"""track_video.py — Use Case B v1: video person tracking + face-ID.

Pipeline:
  1. YOLOv8 person detection + ByteTrack/BoTSORT stable track IDs (GPU via ROCm)
  2. Every FACE_SAMPLE_INTERVAL frames per active track: crop the person box,
     run the YuNet-first face-detect+dlib-encode pipeline (same as engine.py /
     scan_photos.py), compare to enrolled embeddings.
  3. Sticky labelling: once a track is confirmed as "Toby" at <= tolerance,
     that label persists. Reconfirm every FACE_SAMPLE_INTERVAL frames to handle
     track ID reuse and long takes.
  4. Output: per-frame list of track dicts; write to JSON file.

Ticket 6bcb1adc.

Runs in this repo's own .venv — torch, torchvision and ultralytics are
installed there:

    .venv/bin/python track_video.py <video.mp4> --annotated <out.mp4>

IMPORTANT: set HSA_OVERRIDE_GFX_VERSION before any torch import so ROCm
           correctly maps RX 6600 GFX architecture. This module does it below,
           and only on a host that actually has the AMD stack loaded.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# --- ROCm GFX override: must happen before torch / ultralytics import ---
# Only meaningful on an AMD host; setting it elsewhere just misleads a crash log.
if os.path.isdir("/sys/module/amdgpu") or os.path.isdir("/opt/rocm"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import cv2
import numpy as np

# Repo root (so we can import engine.py even when invoked from another directory)
_REPO = Path(__file__).parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import engine as _engine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
YUNET_PATH = _REPO / "models" / "face_detection_yunet.onnx"
DEFAULT_YOLO_MODEL = str(_REPO / "yolov8n.pt")
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_CONFIDENCE = 0.3
DEFAULT_TOLERANCE = 0.50
DEFAULT_FACE_INTERVAL = 30  # frames between face-ID checks per track


# ---------------------------------------------------------------------------
# Track state management
# ---------------------------------------------------------------------------

class TrackState:
    """Mutable state for one track across the video."""

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.is_toby: bool = False
        self.best_face_distance: Optional[float] = None
        self.last_face_check_frame: int = -9999
        self.frame_count: int = 0
        self.conf_sum: float = 0.0

    def update_face(self, is_toby: bool, face_distance: float) -> None:
        if is_toby:
            self.is_toby = True
        self.best_face_distance = (
            face_distance
            if self.best_face_distance is None
            else min(self.best_face_distance, face_distance)
        )

    @property
    def avg_conf(self) -> float:
        return self.conf_sum / max(self.frame_count, 1)


# ---------------------------------------------------------------------------
# Face-ID subsystem
# ---------------------------------------------------------------------------

def _build_face_engine(tolerance: float):
    """Build the YuNet-first face detect+encode engine (same as engine.py)."""
    import face_recognition

    yn = None
    if YUNET_PATH.exists():
        yn = cv2.FaceDetectorYN.create(
            str(YUNET_PATH), "", (320, 320), 0.5, 0.3, 5000
        )

    # Reuse the shared known-encoding cache from engine.py.
    # This guarantees the embedding space is IDENTICAL to enroll.py output —
    # critical so the calibrated 0.50 threshold is valid.
    encs, names = _engine._load_known_encodings(face_recognition)
    return face_recognition, yn, encs, names, tolerance


def _face_id_crop(
    crop_bgr: np.ndarray,
    face_recognition,
    yn,
    encs: list,
    names: list,
    tolerance: float,
) -> tuple[bool, Optional[float]]:
    """
    Run face detect + encode on a BGR crop.
    Returns (is_toby: bool, face_distance: float | None).
    Face distance is None if no face was detected in the crop.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False, None

    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return False, None

    arr_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # YuNet detection (angle-robust) — same order as engine._load_known_encodings
    locs = []
    if yn is not None:
        try:
            yn.setInputSize((w, h))
            _, faces = yn.detect(crop_bgr)
            if faces is not None:
                for f in faces:
                    x, y, bw, bh = (int(v) for v in f[:4])
                    x, y = max(0, x), max(0, y)
                    locs.append((y, x + bw, y + bh, x))  # dlib order
        except Exception:
            locs = []

    # HOG fallback
    if not locs:
        locs = face_recognition.face_locations(arr_rgb, model="hog")

    if not locs:
        return False, None

    fencs = face_recognition.face_encodings(arr_rgb, locs[:1])
    if not fencs:
        return False, None

    _, dist, known = _engine._match(encs, names, fencs[0], tolerance)
    return known, dist


# ---------------------------------------------------------------------------
# Main tracking function
# ---------------------------------------------------------------------------

def track_video(
    source: str | Path,
    output_json: str | Path,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    tracker: str = DEFAULT_TRACKER,
    confidence: float = DEFAULT_CONFIDENCE,
    tolerance: float = DEFAULT_TOLERANCE,
    face_interval: int = DEFAULT_FACE_INTERVAL,
    annotated_output: Optional[str | Path] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Track persons in a video, label Toby by face-ID, write per-frame JSON.

    Args:
        source:           Input video path.
        output_json:      Where to write per-frame tracking JSON.
        yolo_model:       Path to YOLO .pt file.
        tracker:          'bytetrack.yaml' or 'botsort.yaml'.
        confidence:       YOLO detection confidence threshold.
        tolerance:        Face-match tolerance (default 0.50, calibrated).
        face_interval:    Frames between face-ID checks per track.
        annotated_output: Optional path for annotated output video.
        verbose:          Print progress.

    Returns:
        List of per-frame dicts (same as JSON output).
    """
    from ultralytics import YOLO

    source = Path(source)
    output_json = Path(output_json)

    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    # --- Load YOLO model ---
    if verbose:
        print(f"[track] Loading YOLO model: {yolo_model}")
    model = YOLO(yolo_model)

    # --- Verify GPU ---
    import torch
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        if torch.cuda.is_available():
            print(f"[track] GPU: {torch.cuda.get_device_name(0)} ({device_str})")
        else:
            print("[track] WARNING: GPU not available, running on CPU")

    # --- Build face engine ---
    if verbose:
        print("[track] Building face-ID engine (YuNet + dlib)...")
    face_recognition, yn, encs, names, tol = _build_face_engine(tolerance)
    if verbose:
        print(f"[track] Enrolled embeddings: {len(encs)} | people: {sorted(set(names))}")
        print(f"[track] Face tolerance: {tol} | Face-ID interval: {face_interval} frames")

    # --- Video info ---
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if verbose:
        print(f"[track] Video: {source.name} | {w_vid}x{h_vid} @ {fps:.1f}fps | {total_frames} frames")

    # --- Video writer setup ---
    writer = None
    if annotated_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(annotated_output), fourcc, fps, (w_vid, h_vid)
        )

    # --- Track state dict (track_id -> TrackState) ---
    states: dict[int, TrackState] = {}

    # --- Run tracking ---
    all_frames: list[dict] = []
    t_start = time.time()
    frame_idx = 0

    # ultralytics stream=True yields Results objects one per frame
    results = model.track(
        source=str(source),
        tracker=tracker,
        classes=[0],          # class 0 = person
        conf=confidence,
        stream=True,
        device=device_str,
        verbose=False,
    )

    for r in results:
        frame_bgr = r.orig_img  # BGR numpy array
        frame_h, frame_w = frame_bgr.shape[:2]

        tracks_this_frame: list[dict] = []

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()     # pixel coords
            xywhn = r.boxes.xywhn.cpu().numpy()   # normalised cx,cy,w,h
            confs = r.boxes.conf.cpu().numpy()

            for tid, box_xyxy, box_norm, det_conf in zip(ids, xyxy, xywhn, confs):
                tid = int(tid)
                det_conf = float(det_conf)

                # Ensure state exists
                if tid not in states:
                    states[tid] = TrackState(tid)
                st = states[tid]
                st.frame_count += 1
                st.conf_sum += det_conf

                # --- Face-ID check ---
                face_distance: Optional[float] = None
                do_face_check = (
                    encs and
                    (frame_idx - st.last_face_check_frame) >= face_interval
                )
                if do_face_check:
                    st.last_face_check_frame = frame_idx
                    # Crop upper ~50% of person box for the head region
                    x1, y1, x2, y2 = (int(v) for v in box_xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_w, x2), min(frame_h, y2)
                    mid_y = y1 + max(1, (y2 - y1) // 2)
                    crop = frame_bgr[y1:mid_y, x1:x2]
                    try:
                        is_match, fd = _face_id_crop(
                            crop, face_recognition, yn, encs, names, tol
                        )
                    except Exception as exc:
                        is_match, fd = False, None
                        if verbose:
                            print(f"[track] face-ID error track {tid} frame {frame_idx}: {exc}")
                    st.update_face(is_match, fd if fd is not None else 1.0)
                    face_distance = fd

                tracks_this_frame.append({
                    "id": tid,
                    "box_xywh_norm": [round(float(v), 4) for v in box_norm],
                    "is_toby": st.is_toby,
                    "conf": round(det_conf, 3),
                    "face_distance": round(face_distance, 4) if face_distance is not None else None,
                })

                # --- Annotate frame ---
                if writer is not None:
                    x1, y1, x2, y2 = (int(v) for v in box_xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_w, x2), min(frame_h, y2)
                    color = (0, 200, 0) if st.is_toby else (180, 180, 180)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    label = f"{'Toby' if st.is_toby else 'ID'} #{tid}  {det_conf:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(frame_bgr, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
                    cv2.putText(
                        frame_bgr, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA
                    )

        all_frames.append({"frame": frame_idx, "tracks": tracks_this_frame})

        if writer is not None:
            writer.write(frame_bgr)

        frame_idx += 1
        if verbose and frame_idx % 30 == 0:
            elapsed = time.time() - t_start
            fps_actual = frame_idx / max(elapsed, 0.001)
            pct = 100 * frame_idx / max(total_frames, 1)
            print(
                f"\r[track] {frame_idx}/{total_frames} ({pct:.0f}%) | "
                f"{fps_actual:.1f} it/s | {elapsed:.1f}s elapsed",
                end="", flush=True,
            )

    if writer is not None:
        writer.release()

    elapsed = time.time() - t_start
    fps_actual = frame_idx / max(elapsed, 0.001)
    if verbose:
        print(f"\n[track] Done: {frame_idx} frames in {elapsed:.1f}s ({fps_actual:.1f} it/s)")
        toby_tracks = [tid for tid, st in states.items() if st.is_toby]
        print(f"[track] Tracks found: {len(states)} | Toby tracks: {toby_tracks}")

    # --- Write JSON ---
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(all_frames, f, separators=(",", ":"))
    if verbose:
        print(f"[track] JSON written: {output_json} ({output_json.stat().st_size // 1024} KB)")

    return all_frames


# ---------------------------------------------------------------------------
# Track summary helper (used by Gradio UI)
# ---------------------------------------------------------------------------

def summarize_tracks(frames: list[dict]) -> list[dict]:
    """Collapse per-frame data into one summary row per track."""
    from collections import defaultdict
    totals: dict[int, dict] = {}
    for fr in frames:
        for t in fr["tracks"]:
            tid = t["id"]
            if tid not in totals:
                totals[tid] = {"id": tid, "is_toby": False, "frames_present": 0, "conf_sum": 0.0}
            totals[tid]["frames_present"] += 1
            totals[tid]["conf_sum"] += t["conf"]
            if t["is_toby"]:
                totals[tid]["is_toby"] = True

    summary = []
    for tid in sorted(totals):
        d = totals[tid]
        n = d["frames_present"]
        summary.append({
            "track_id": tid,
            "is_toby": d["is_toby"],
            "frames_present": n,
            "avg_conf": round(d["conf_sum"] / max(n, 1), 3),
        })
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Track persons in a video and label Toby by face-ID.",
    )
    parser.add_argument("source", type=Path, help="Input video file")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Per-frame JSON output path (default: <source>.tracks.json)")
    parser.add_argument("--annotated", type=Path, default=None,
                        help="Annotated video output path")
    parser.add_argument("--model", default=DEFAULT_YOLO_MODEL,
                        help=f"YOLO model path (default: {DEFAULT_YOLO_MODEL})")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER,
                        choices=["bytetrack.yaml", "botsort.yaml"],
                        help=f"Tracker config (default: {DEFAULT_TRACKER})")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"Detection confidence threshold (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help=f"Face-match tolerance (default: {DEFAULT_TOLERANCE})")
    parser.add_argument("--face-interval", type=int, default=DEFAULT_FACE_INTERVAL,
                        help=f"Frames between face-ID checks per track (default: {DEFAULT_FACE_INTERVAL})")
    args = parser.parse_args()

    out_json = args.output_json or args.source.with_suffix(".tracks.json")
    frames = track_video(
        source=args.source,
        output_json=out_json,
        yolo_model=args.model,
        tracker=args.tracker,
        confidence=args.confidence,
        tolerance=args.tolerance,
        face_interval=args.face_interval,
        annotated_output=args.annotated,
        verbose=True,
    )
    summary = summarize_tracks(frames)
    print("\n=== Track Summary ===")
    for row in summary:
        label = "TOBY" if row["is_toby"] else "unknown"
        print(f"  Track {row['track_id']:3d}: {label:7s}  "
              f"{row['frames_present']:4d} frames  avg_conf={row['avg_conf']:.3f}")


if __name__ == "__main__":
    main()
