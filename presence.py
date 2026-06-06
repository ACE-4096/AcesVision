"""Presence automation — turn "who's in which room" into enter/leave events.

Runs the cameras headlessly (no GUI), recognises faces, and tracks who is in
each room (one camera = one room). On every enter/leave it:
  - logs to presence.db (SQLite),
  - fires any matching rules in automations.json (webhook or shell command).

    python presence.py                 # uses cameras.json + automations.json

Rooms are the camera "name" fields in cameras.json.

automations.json is a list of rules; each fires on a matching event:
    [
      { "event": "enter", "person": "Toby", "room": "Office",
        "webhook": { "url": "http://homeassistant.local/api/webhook/office_on", "method": "POST" } },
      { "event": "leave", "room": "Office",
        "webhook": { "url": "http://homeassistant.local/api/webhook/office_off" } },
      { "event": "enter", "person": "Emma",
        "command": "notify-send 'Emma is in the {room}'" }
    ]
person/room may be omitted or "*" to match any. Command/webhook strings get
{person} {room} {event} {time} substituted.

Env: FACE_ID_LEAVE_AFTER (s, default 8), FACE_ID_MIN_SIGHTINGS (default 2),
     FACE_ID_DETECT_EVERY (default 5), plus engine vars.
"""
import json
import os
import sqlite3
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2

import roomview  # reuse open_source + load_config
from engine import build_detector

HERE = Path(__file__).parent
DB_PATH = HERE / "presence.db"
LEAVE_AFTER = float(os.environ.get("FACE_ID_LEAVE_AFTER", "8"))
MIN_SIGHTINGS = int(os.environ.get("FACE_ID_MIN_SIGHTINGS", "2"))
DETECT_EVERY = int(os.environ.get("FACE_ID_DETECT_EVERY", "5"))


class PresenceTracker:
    """Debounced per-(person, room) presence. Time is passed in for testability.

    enter fires after MIN_SIGHTINGS sightings; leave fires after LEAVE_AFTER
    seconds with no sighting. on_event(event, person, room, ts) is the callback.
    """

    def __init__(self, leave_after=LEAVE_AFTER, min_sightings=MIN_SIGHTINGS,
                 on_event=None):
        self.leave_after = leave_after
        self.min_sightings = min_sightings
        self.on_event = on_event
        self.last_seen = {}      # key -> ts
        self.present = set()     # keys currently "in the room"
        self._pending = {}       # key -> sighting count (not yet present)
        self._lock = threading.Lock()

    def observe(self, person, room, now):
        key = (person, room)
        with self._lock:
            self.last_seen[key] = now
            if key in self.present:
                return
            self._pending[key] = self._pending.get(key, 0) + 1
            if self._pending[key] >= self.min_sightings:
                self._pending.pop(key, None)
                self.present.add(key)
                fire = True
            else:
                fire = False
        if fire:
            self._emit("enter", person, room, now)

    def tick(self, now):
        leaving = []
        with self._lock:
            for key in list(self.present):
                if now - self.last_seen.get(key, 0) > self.leave_after:
                    self.present.discard(key)
                    leaving.append(key)
            # expire stale pending sightings that never became present
            for key in list(self._pending):
                if now - self.last_seen.get(key, 0) > self.leave_after:
                    self._pending.pop(key, None)
        for person, room in leaving:
            self._emit("leave", person, room, now)

    def in_room(self):
        with self._lock:
            return sorted(self.present)

    def _emit(self, event, person, room, now):
        if self.on_event:
            self.on_event(event, person, room, now)


# --- side effects: logging + actions ---------------------------------------
def init_db():
    # check_same_thread=False: enter events fire from worker threads, leave from
    # the main loop — all DB writes are serialised by _DB_LOCK below.
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("CREATE TABLE IF NOT EXISTS events "
                "(ts TEXT, epoch REAL, person TEXT, room TEXT, event TEXT)")
    con.commit()
    return con


_DB_LOCK = threading.Lock()


def load_rules():
    path = HERE / "automations.json"
    if not path.exists():
        print("[automations] no automations.json — events are logged only.")
        return []
    rules = json.load(open(path))
    print(f"[automations] {len(rules)} rule(s) loaded")
    return rules


def _matches(rule, ev):
    """ev is a dict {event, person?, room?, gesture?}. A rule constrains only
    the fields it names; '*' or omission matches anything."""
    if rule.get("event") != ev.get("event"):
        return False
    for field in ("person", "room", "gesture"):
        if field in rule and rule[field] not in ("*", ev.get(field)):
            return False
    return True


def fire_rules(rules, ev):
    fields = {"person": ev.get("person", ""), "room": ev.get("room", ""),
              "event": ev.get("event", ""), "gesture": ev.get("gesture", ""),
              "time": datetime.now().strftime("%H:%M:%S")}
    for rule in rules:
        if not _matches(rule, ev):
            continue
        if "webhook" in rule:
            wh = rule["webhook"]
            url = wh["url"].format(**fields)
            method = wh.get("method", "GET").upper()
            try:
                data = json.dumps(fields).encode() if method == "POST" else None
                req = urllib.request.Request(url, data=data, method=method,
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5)
                print(f"    webhook {method} {url} -> ok")
            except Exception as e:
                print(f"    webhook {url} FAILED: {e}")
        if "command" in rule:
            cmd = rule["command"].format(**fields)
            try:
                subprocess.Popen(cmd, shell=True)
                print(f"    ran: {cmd}")
            except Exception as e:
                print(f"    command FAILED: {e}")


# --- live capture per camera ------------------------------------------------
class Worker(threading.Thread):
    def __init__(self, spec, tracker):
        super().__init__(daemon=True)
        self.spec = spec
        self.room = spec.get("name", "room")
        self.detector = build_detector()
        self.tracker = tracker
        self.running = True

    def run(self):
        cap = roomview.open_source(self.spec)
        i = 0
        while self.running:
            if cap is None or not cap.isOpened():
                time.sleep(1.0)
                cap = roomview.open_source(self.spec)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                time.sleep(0.5)
                cap = roomview.open_source(self.spec)
                continue
            if i % DETECT_EVERY == 0:
                now = time.monotonic()
                try:
                    faces = self.detector(frame)
                except Exception as e:
                    faces = []
                    print(f"[{self.room}] detect error: {e}")
                for f in faces:                 # outside try so callback errors surface
                    if f.known:
                        self.tracker.observe(f.name, self.room, now)
            i += 1
        if cap is not None:
            cap.release()


def main():
    con = init_db()
    rules = load_rules()

    def on_event(event, person, room, now):
        stamp = datetime.now()
        arrow = "->" if event == "enter" else "<-"
        print(f"[{stamp.strftime('%H:%M:%S')}] {person} {arrow} {room} ({event})")
        with _DB_LOCK:
            con.execute("INSERT INTO events VALUES (?,?,?,?,?)",
                        (stamp.isoformat(), now, person, room, event))
            con.commit()
        fire_rules(rules, {"event": event, "person": person, "room": room})

    tracker = PresenceTracker(on_event=on_event)
    specs = roomview.load_config()
    workers = [Worker(s, tracker) for s in specs]
    print(f"[engine] {workers[0].detector.engine} | "
          f"watching rooms: {', '.join(w.room for w in workers)}")
    for w in workers:
        w.start()

    try:
        while True:
            time.sleep(1.0)
            tracker.tick(time.monotonic())
            here = tracker.in_room()
            print(f"  in rooms: {', '.join(f'{p}@{r}' for p, r in here) or '(nobody)'}",
                  end="\r")
    except KeyboardInterrupt:
        print("\n[stop]")
        for w in workers:
            w.running = False


if __name__ == "__main__":
    main()
