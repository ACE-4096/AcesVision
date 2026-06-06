"""Web rep counter — pose-based rep counting streamed to the browser.

Runs the camera + MediaPipe Pose, counts reps, and serves a dashboard with the
live annotated feed, a big rep count, the working-joint angle bar, exercise
switch buttons, and a reset. No local OpenCV window (which clashes with
MediaPipe's GL context) — everything is in the browser.

    python reps_server.py                 # open http://localhost:8000
    python reps_server.py --host 0.0.0.0  # reach from your phone at the gym
    FACE_ID_EXERCISE=squat python reps_server.py
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify, request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import camera
import reps

HERE = Path(__file__).parent
JPEG_Q = int(os.environ.get("FACE_ID_JPEG_Q", "92"))
# Detect pose on a downscaled copy (landmarks are normalised, so they map back
# to full res) — keeps the stream sharp at 1080p without the full-frame pose cost.
POSE_W = int(os.environ.get("FACE_ID_POSE_W", "960"))


class RepFeed(threading.Thread):
    def __init__(self, ex_key):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.jpeg = None
        self.status = "connecting"
        self.running = True
        self.angle = None
        self.landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=reps.MODEL),
                num_poses=1))
        self.set_exercise(ex_key)

    def set_exercise(self, key):
        if key not in reps.EXERCISES:
            return
        with self.lock:
            self.ex_key = key
            self.ex = reps.EXERCISES[key]
            self.counter = reps.RepCounter(self.ex.rest, self.ex.active)

    def reset(self):
        with self.lock:
            self.counter.count = 0

    def run(self):
        cap = camera.open_camera()
        flash, backoff = 0, 1.0
        while self.running:
            if cap is None or not cap.isOpened():
                self.status = "reconnecting"
                time.sleep(backoff)
                backoff = min(backoff * 2, 15)
                try:
                    cap = camera.open_camera()
                except RuntimeError:
                    cap = None
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                self.status = "reconnecting"
                cap.release()
                cap = None
                continue
            backoff = 1.0
            self.status = "live"
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (POSE_W, int(h * POSE_W / w))) if w > POSE_W else frame
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res = self.landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            pts = reps.landmarks_px(res, w, h)   # normalised -> full-res pixels
            with self.lock:
                ex, counter = self.ex, self.counter
            angle = reps.joint_angle(pts, ex.joints) if pts else None
            if angle is not None and counter.update(angle):
                flash = 6
            reps.draw(frame, pts, ex, counter, angle, flash > 0)
            flash = max(0, flash - 1)
            self.angle = angle
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if ok:
                with self.lock:
                    self.jpeg = buf.tobytes()
        if cap is not None:
            cap.release()

    def state(self):
        with self.lock:
            a = self.angle
            return {"exercise": self.ex.name, "key": self.ex_key,
                    "count": self.counter.count, "stage": self.counter.stage,
                    "angle": None if a is None else round(a),
                    "progress": round(self.counter.progress(a), 2) if a is not None else 0,
                    "status": self.status}


app = Flask(__name__)
FEED = None

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Rep Counter</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#0d1117;--card:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#8b949e;--accent:#2f81f7;--ok:#2ea043}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center}
 header h1{font-size:17px;margin:0}.dot{width:9px;height:9px;border-radius:50%;background:var(--ok);box-shadow:0 0 8px var(--ok)}
 .wrap{display:flex;gap:18px;padding:18px;flex-wrap:wrap}
 .feed{flex:2;min-width:340px;background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}
 .feed img{width:100%;display:block;background:#000;aspect-ratio:16/9;object-fit:contain}
 .side{flex:1;min-width:260px;display:flex;flex-direction:column;gap:16px}
 .count{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;text-align:center}
 .count .n{font-size:108px;font-weight:800;line-height:1;transition:color .1s}
 .count .n.flash{color:var(--ok)}
 .count .ex{color:var(--mut);text-transform:uppercase;letter-spacing:.08em;font-size:13px}
 .count .stage{margin-top:8px;font-weight:600}.stage.active{color:var(--accent)}
 .bar{height:14px;background:#21262d;border-radius:8px;overflow:hidden;margin-top:14px}
 .bar>div{height:100%;background:var(--accent);width:0%;transition:width .08s}
 .btns{display:flex;flex-wrap:wrap;gap:8px}
 button{flex:1;min-width:72px;background:#21262d;color:var(--fg);border:1px solid var(--line);
   border-radius:10px;padding:10px;font-size:14px;cursor:pointer}
 button.on{background:var(--accent);border-color:var(--accent)}
 button.reset{background:#3d1418;border-color:#5a1f25}
</style></head><body>
<header><span class=dot></span><h1>Rep Counter</h1><span style="color:var(--mut)" id=st></span></header>
<div class=wrap>
 <div class=feed><img src="/stream"></div>
 <div class=side>
  <div class=count>
    <div class=ex id=ex>—</div>
    <div class="n" id=n>0</div>
    <div class=stage id=stage>—</div>
    <div class=bar><div id=fill></div></div>
    <div style="color:var(--mut);font-size:13px;margin-top:8px" id=angle></div>
  </div>
  <div class=btns id=exbtns></div>
  <div class=btns><button class=reset onclick="fetch('/reset',{method:'POST'})">Reset</button></div>
 </div>
</div>
<script>
const EX={curl:"Curl",squat:"Squat",pushup:"Push-up",press:"Press"};
const eb=document.getElementById('exbtns');
Object.keys(EX).forEach(k=>{const b=document.createElement('button');b.textContent=EX[k];
 b.dataset.k=k;b.onclick=()=>fetch('/set?ex='+k);eb.appendChild(b);});
let prev=0;
async function poll(){
 try{const s=await (await fetch('/api/reps')).json();
  document.getElementById('ex').textContent=s.exercise;
  const n=document.getElementById('n');n.textContent=s.count;
  if(s.count>prev){n.classList.add('flash');setTimeout(()=>n.classList.remove('flash'),200);}
  prev=s.count;
  const st=document.getElementById('stage');st.textContent=s.stage.toUpperCase();
  st.className='stage '+(s.stage==='active'?'active':'');
  document.getElementById('fill').style.width=(s.progress*100)+'%';
  document.getElementById('angle').textContent=s.angle!=null?('joint angle '+s.angle+'°'):'step into frame';
  document.getElementById('st').textContent=s.status;
  document.querySelectorAll('#exbtns button').forEach(b=>b.classList.toggle('on',b.dataset.k===s.key));
 }catch(e){}
}
setInterval(poll,200);poll();
</script></body></html>"""


@app.route("/")
def index():
    return PAGE


@app.route("/stream")
def stream():
    def gen():
        import numpy as np
        blank = None
        while True:
            with FEED.lock:
                jpg = FEED.jpeg
            if jpg is None:
                if blank is None:
                    img = np.zeros((360, 640, 3), "uint8")
                    cv2.putText(img, f"{FEED.status}...", (20, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
                    blank = cv2.imencode(".jpg", img)[1].tobytes()
                jpg = blank
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/reps")
def api_reps():
    return jsonify(FEED.state())


@app.route("/reset", methods=["POST"])
def reset():
    FEED.reset()
    return ("", 204)


@app.route("/set")
def set_ex():
    FEED.set_exercise(request.args.get("ex", "curl"))
    return ("", 204)


def main():
    global FEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("exercise", nargs="?", default=os.environ.get("FACE_ID_EXERCISE", "curl"))
    args = ap.parse_args()

    FEED = RepFeed(args.exercise)
    FEED.start()
    print(f"[reps] exercise={args.exercise}  open http://{args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        FEED.running = False
        time.sleep(0.6)


if __name__ == "__main__":
    main()
