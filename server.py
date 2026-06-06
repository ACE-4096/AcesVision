"""Web dashboard — the cool GUI.

Runs the cameras, recognises faces AND hand gestures, and serves a browser
dashboard with live annotated feeds, a "who's in the room" panel, and a live
event feed. Presence (enter/leave) and gestures both fire automations.json rules.

    python server.py            # then open http://localhost:8000
    python server.py --port 8080 --host 0.0.0.0   # reachable from your phone

Gestures (MediaPipe, built-in): Thumb_Up Thumb_Down Victory Open_Palm
Closed_Fist Pointing_Up ILoveYou. Map them to actions in automations.json:
    { "event": "gesture", "gesture": "Thumb_Up", "room": "Office",
      "webhook": { "url": "http://homeassistant.local/api/webhook/toggle_lights" } }
"""
import argparse
import json
import queue
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify

import roomview
from engine import build_detector
from gestures import GestureDetector
from presence import PresenceTracker, fire_rules, load_rules

HERE = Path(__file__).parent
DB_PATH = HERE / "events.db"
DETECT_EVERY = 5
GESTURE_COOLDOWN = 2.5           # seconds before the same gesture re-fires
JPEG_Q = int(os.environ.get("FACE_ID_JPEG_Q", "92"))
GREEN, RED, BLUE = (0, 200, 0), (40, 40, 220), (230, 160, 30)
EMOJI = {"Thumb_Up": "👍", "Thumb_Down": "👎", "Victory": "✌️",
         "Open_Palm": "🖐️", "Closed_Fist": "✊", "Pointing_Up": "☝️",
         "ILoveYou": "🤟"}


class Hub:
    """Central event bus: presence + gestures -> db, rules, SSE, recent list."""

    def __init__(self):
        self.rules = load_rules()
        self.tracker = PresenceTracker(on_event=self._on_presence)
        self.recent = deque(maxlen=200)
        self.subs = []
        self.lock = threading.Lock()
        self.con = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS events "
                         "(ts TEXT, epoch REAL, kind TEXT, person TEXT, room TEXT, detail TEXT)")
        self.con.commit()
        threading.Thread(target=self._tick_loop, daemon=True).start()

    def _tick_loop(self):
        while True:
            time.sleep(1.0)
            self.tracker.tick(time.monotonic())

    def _on_presence(self, event, person, room, now):
        self.emit({"event": event, "person": person, "room": room})

    def emit(self, ev):
        ev = dict(ev)
        ev["time"] = datetime.now().strftime("%H:%M:%S")
        kind = ev["event"]
        detail = ev.get("gesture", "")
        with self.lock:
            self.con.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",
                             (datetime.now().isoformat(), time.monotonic(), kind,
                              ev.get("person", ""), ev.get("room", ""), detail))
            self.con.commit()
            self.recent.appendleft(ev)
            subs = list(self.subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass
        fire_rules(self.rules, ev)

    def rooms(self):
        out = {}
        for person, room in self.tracker.in_room():
            out.setdefault(room, []).append(person)
        return out

    def subscribe(self):
        q = queue.Queue(maxsize=50)
        with self.lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)


class CamFeed(threading.Thread):
    def __init__(self, spec, hub):
        super().__init__(daemon=True)
        self.spec = spec
        self.room = spec.get("name", "cam")
        self.hub = hub
        self.face = build_detector()
        self.gest = GestureDetector()
        self.lock = threading.Lock()
        self.jpeg = None
        self.status = "connecting"
        self.running = True
        self._gesture_at = {}

    def _annotate(self, frame, faces, gests):
        for f in faces:
            c = GREEN if f.known else RED
            label = f"{f.name} {f.conf:.2f}" if f.known else "Unknown"
            cv2.rectangle(frame, (f.x, f.y), (f.x + f.w, f.y + f.h), c, 2)
            cv2.putText(frame, label, (f.x, max(f.y - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        for g in gests:
            cv2.rectangle(frame, (g.x, g.y), (g.x + g.w, g.y + g.h), BLUE, 2)
            cv2.putText(frame, g.name, (g.x, max(g.y - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2)
        return frame

    def run(self):
        cap = roomview.open_source(self.spec)
        i, faces, gests = 0, [], []
        backoff = 1.0
        while self.running:
            if cap is None or not cap.isOpened():
                self.status = "reconnecting"
                time.sleep(backoff)
                backoff = min(backoff * 2, 15)   # back off — don't hammer a missing cam
                cap = roomview.open_source(self.spec)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                self.status = "reconnecting"
                cap.release()
                cap = None
                time.sleep(backoff)
                backoff = min(backoff * 2, 15)
                continue
            backoff = 1.0                        # reset on a good read
            self.status = "live"
            if i % DETECT_EVERY == 0:
                now = time.monotonic()
                try:
                    faces = self.face(frame)
                except Exception:
                    faces = []
                try:
                    gests = self.gest.detect(frame)
                except Exception:
                    gests = []
                known_here = [f.name for f in faces if f.known]
                for f in faces:
                    if f.known:
                        self.hub.tracker.observe(f.name, self.room, now)
                for g in gests:
                    if now - self._gesture_at.get(g.name, 0) > GESTURE_COOLDOWN:
                        self._gesture_at[g.name] = now
                        who = known_here[0] if known_here else "someone"
                        self.hub.emit({"event": "gesture", "gesture": g.name,
                                       "room": self.room, "person": who})
            self._annotate(frame, faces, gests)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if ok:
                with self.lock:
                    self.jpeg = buf.tobytes()
            i += 1
        if cap is not None:
            cap.release()

    def frame_bytes(self):
        with self.lock:
            return self.jpeg


app = Flask(__name__)
HUB = None
FEEDS = {}

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Face ID — Room View</title>
<meta name=viewport content="width=device-width, initial-scale=1">
<style>
 :root{--bg:#0d1117;--card:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#8b949e;--accent:#2f81f7}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
   align-items:center;gap:10px} header h1{font-size:16px;margin:0;font-weight:600}
 .dot{width:9px;height:9px;border-radius:50%;background:#2ea043;box-shadow:0 0 8px #2ea043}
 .wrap{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
 .feeds{flex:3;min-width:320px;display:grid;gap:14px;
   grid-template-columns:repeat(auto-fit,minmax(360px,1fr))}
 .feed{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
 .feed h2{margin:0;padding:8px 12px;font-size:13px;font-weight:600;
   display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--line)}
 .feed img{width:100%;display:block;background:#000;aspect-ratio:16/9;object-fit:contain}
 .side{flex:1;min-width:260px;display:flex;flex-direction:column;gap:16px}
 .panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
 .panel h3{margin:0 0 10px;font-size:13px;color:var(--mut);text-transform:uppercase;
   letter-spacing:.04em}
 .room{margin-bottom:10px} .room b{color:var(--accent)}
 .chip{display:inline-block;background:#21262d;border:1px solid var(--line);
   border-radius:20px;padding:2px 10px;margin:2px 4px 2px 0;font-size:13px}
 #events{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;max-height:60vh;overflow:auto}
 .ev{padding:5px 0;border-bottom:1px solid #21262d;display:flex;gap:8px}
 .ev .t{color:var(--mut)} .ev.enter .a{color:#2ea043}.ev.leave .a{color:#f85149}
 .ev.gesture .a{color:var(--accent)} .empty{color:var(--mut)}
</style></head><body>
<header><span class=dot></span><h1>Face ID — Room View</h1>
  <span style="color:var(--mut)" id=engine></span></header>
<div class=wrap>
  <div class=feeds id=feeds></div>
  <div class=side>
    <div class=panel><h3>In the room</h3><div id=rooms class=empty>nobody</div></div>
    <div class=panel><h3>Live events</h3><div id=events class=empty>waiting…</div></div>
  </div>
</div>
<script>
const EMOJI={Thumb_Up:"👍",Thumb_Down:"👎",Victory:"✌️",Open_Palm:"🖐️",
  Closed_Fist:"✊",Pointing_Up:"☝️",ILoveYou:"🤟"};
const feeds=__FEEDS__;
const fc=document.getElementById('feeds');
feeds.forEach(n=>{fc.insertAdjacentHTML('beforeend',
  `<div class=feed><h2><span class=dot></span>${n}</h2>`+
  `<img src="/stream/${encodeURIComponent(n)}" alt="${n}"></div>`)});

async function poll(){
  try{const s=await (await fetch('/api/state')).json();
    const r=document.getElementById('rooms');
    const ks=Object.keys(s.rooms);
    r.className=ks.length?'':'empty';
    r.innerHTML=ks.length?ks.map(k=>`<div class=room><b>${k}</b>: `+
      s.rooms[k].map(p=>`<span class=chip>${p}</span>`).join('')+`</div>`).join('')
      :'nobody';
  }catch(e){}
}
setInterval(poll,2000); poll();

const log=document.getElementById('events'); let first=true;
const es=new EventSource('/api/events');
es.onmessage=e=>{
  const v=JSON.parse(e.data);
  if(first){log.innerHTML='';log.className='';first=false;}
  let txt;
  if(v.event==='gesture') txt=`${EMOJI[v.gesture]||'✋'} ${v.gesture} by ${v.person} @ ${v.room}`;
  else txt=`${v.event==='enter'?'→':'←'} ${v.person} ${v.event==='enter'?'entered':'left'} ${v.room}`;
  log.insertAdjacentHTML('afterbegin',
    `<div class="ev ${v.event}"><span class=t>${v.time}</span><span class=a>${txt}</span></div>`);
};
</script></body></html>"""


@app.route("/")
def index():
    return PAGE.replace("__FEEDS__", json.dumps(list(FEEDS.keys())))


@app.route("/stream/<name>")
def stream(name):
    feed = FEEDS.get(name)
    if feed is None:
        return "no such camera", 404

    def gen():
        blank = None
        while True:
            jpg = feed.frame_bytes()
            if jpg is None:
                if blank is None:
                    import numpy as np
                    img = np.zeros((360, 640, 3), "uint8")
                    cv2.putText(img, f"{feed.status}...", (20, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
                    blank = cv2.imencode(".jpg", img)[1].tobytes()
                jpg = blank
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def state():
    return jsonify(rooms=HUB.rooms(), events=list(HUB.recent)[:30])


@app.route("/api/events")
def events():
    def gen():
        q = HUB.subscribe()
        try:
            yield "retry: 3000\n\n"
            while True:
                try:
                    ev = q.get(timeout=15)
                    yield f"data: {json.dumps(ev)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            HUB.unsubscribe(q)
    return Response(gen(), mimetype="text/event-stream")


def main():
    global HUB
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    HUB = Hub()
    specs = json.load(open(HERE / "cameras.json"))
    for spec in specs:
        feed = CamFeed(spec, HUB)
        FEEDS[feed.room] = feed
        feed.start()
    print(f"[server] {len(FEEDS)} camera(s): {', '.join(FEEDS)}")
    print(f"[server] open http://{args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[server] releasing cameras...")
        for f in FEEDS.values():
            f.running = False
        time.sleep(0.8)   # let worker threads release their captures cleanly


if __name__ == "__main__":
    main()
