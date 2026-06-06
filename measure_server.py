"""Web app — find objects and measure them, live in the browser.

Each camera runs YOLO object detection + ArUco-marker scaling, so the dashboard
shows live feeds with every object outlined and its real-world dimensions, plus
a measurements panel.

    python make_marker.py                 # print marker.png at 100% scale first
    python measure_server.py              # open http://localhost:8000
    python measure_server.py --host 0.0.0.0 --port 8080   # reach from another device
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify

import roomview
from objects import FaceMeasurer, ObjectMeasurer

HERE = Path(__file__).parent
DETECT_EVERY = 3
JPEG_Q = int(os.environ.get("FACE_ID_JPEG_Q", "92"))
# 'face' = measure your face/head (YuNet). 'objects' = measure any object (YOLO).
MODE = os.environ.get("FACE_ID_MEASURE_MODE", "face").lower()


def make_measurer():
    return FaceMeasurer() if MODE == "face" else ObjectMeasurer()


class MeasureFeed(threading.Thread):
    def __init__(self, spec):
        super().__init__(daemon=True)
        self.spec = spec
        self.name = spec.get("name", "cam")
        self.measurer = make_measurer()
        self.lock = threading.Lock()
        self.jpeg = None
        self.objs = []
        self.scale = None
        self.status = "connecting"
        self.running = True

    def run(self):
        cap = roomview.open_source(self.spec)
        i, objs, scale = 0, [], None
        backoff = 1.0
        while self.running:
            if cap is None or not cap.isOpened():
                self.status = "reconnecting"
                time.sleep(backoff)
                backoff = min(backoff * 2, 15)
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
            backoff = 1.0
            self.status = "live"
            if i % DETECT_EVERY == 0:
                try:
                    objs, scale = self.measurer.detect(frame)
                except Exception as e:
                    objs, scale = [], None
                    print(f"[{self.name}] detect error: {e}")
            self.measurer.annotate(frame, objs, scale)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if ok:
                with self.lock:
                    self.jpeg = buf.tobytes()
                    self.objs = objs
                    self.scale = scale
            i += 1
        if cap is not None:
            cap.release()

    def snapshot(self):
        with self.lock:
            return self.jpeg, list(self.objs), self.scale, self.status


app = Flask(__name__)
FEEDS = {}

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Find & Measure</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#0d1117;--card:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#8b949e;--ok:#2ea043;--warn:#d29922}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
   font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center}
 header h1{font-size:16px;margin:0}.dot{width:9px;height:9px;border-radius:50%;background:var(--ok);box-shadow:0 0 8px var(--ok)}
 .wrap{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
 .feeds{flex:3;min-width:320px;display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(380px,1fr))}
 .feed{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
 .feed h2{margin:0;padding:8px 12px;font-size:13px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center}
 .feed img{width:100%;display:block;background:#000;aspect-ratio:16/9;object-fit:contain}
 .side{flex:1;min-width:280px}
 .panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:16px}
 .panel h3{margin:0 0 6px;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
 .scale{font-size:12px;margin-bottom:10px}.scale.no{color:var(--warn)}.scale.ok{color:var(--ok)}
 table{width:100%;border-collapse:collapse;font-size:13px}
 td,th{text-align:left;padding:5px 4px;border-bottom:1px solid #21262d}
 th{color:var(--mut);font-weight:600}.dim{font-family:ui-monospace,Menlo,monospace;color:var(--fg)}
 .empty{color:var(--mut)}
</style></head><body>
<header><span class=dot></span><h1>Find &amp; Measure</h1>
 <span style="color:var(--mut)">live face/head sizing · ArUco scale</span></header>
<div class=wrap>
 <div class=feeds id=feeds></div>
 <div class=side id=side></div>
</div>
<script>
const feeds=__FEEDS__;
const fc=document.getElementById('feeds');
feeds.forEach(n=>fc.insertAdjacentHTML('beforeend',
 `<div class=feed><h2><span class=dot></span>${n}</h2><img src="/stream/${encodeURIComponent(n)}"></div>`));
async function poll(){
 try{const s=await (await fetch('/api/objects')).json();
  document.getElementById('side').innerHTML=feeds.map(n=>{
   const d=s[n]||{objects:[],scale:null};
   const sc=d.scale? `<div class="scale ok">scale ${d.scale.toFixed(2)} px/mm — measuring</div>`
     : `<div class="scale no">no marker in view — sizes disabled</div>`;
   const rows=d.objects.length? d.objects.map(o=>`<tr><td>${o.name}</td>`+
     `<td class=dim>${o.w_mm!=null? Math.max(o.w_mm,o.h_mm).toFixed(0)+' × '+Math.min(o.w_mm,o.h_mm).toFixed(0)+' mm':'—'}</td>`+
     `<td class=dim>${(o.conf*100).toFixed(0)}%</td></tr>`).join('')
     : `<tr><td colspan=3 class=empty>no objects</td></tr>`;
   return `<div class=panel><h3>${n}</h3>${sc}`+
     `<table><tr><th>object</th><th>size</th><th>conf</th></tr>${rows}</table></div>`;
  }).join('');
 }catch(e){}
}
setInterval(poll,700); poll();
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
        import numpy as np
        blank = None
        while True:
            jpg = feed.snapshot()[0]
            if jpg is None:
                if blank is None:
                    img = np.zeros((360, 640, 3), "uint8")
                    cv2.putText(img, f"{feed.status}...", (20, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
                    blank = cv2.imencode(".jpg", img)[1].tobytes()
                jpg = blank
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/objects")
def api_objects():
    out = {}
    for name, feed in FEEDS.items():
        _, objs, scale, _ = feed.snapshot()
        out[name] = {"scale": scale,
                     "objects": [{"name": o.name, "conf": o.conf,
                                  "w_mm": o.w_mm, "h_mm": o.h_mm} for o in objs]}
    return jsonify(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    specs = json.load(open(HERE / "cameras.json"))
    for spec in specs:
        feed = MeasureFeed(spec)
        FEEDS[feed.name] = feed
        feed.start()
    print(f"[measure] {len(FEEDS)} camera(s): {', '.join(FEEDS)}")
    print(f"[measure] open http://{args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        for f in FEEDS.values():
            f.running = False
        time.sleep(0.8)


if __name__ == "__main__":
    main()
