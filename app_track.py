"""app_track.py — Gradio localhost UI for video person-tracking (Use Case B v1).

Ticket cdd38f26.

REQUIRES AN EXTRA DEPENDENCY. Gradio is not in requirements.txt and is not
installed by default; everything else this module needs is already in .venv:

    .venv/bin/pip install gradio
    .venv/bin/python app_track.py

URL: http://127.0.0.1:7860

No data leaves the machine. All inference is local (ROCm GPU for YOLO,
CPU for dlib encoding). Biometric data never written to Gradio tmp.

The launch command here used to name a ROCm venv on an external drive. That
drive is gone and that venv no longer exists; this repo's own .venv carries
torch, torchvision and ultralytics now.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# --- ROCm GFX override: must be set before torch is imported anywhere ---
# Only meaningful on an AMD host; setting it elsewhere just misleads a crash log.
if os.path.isdir("/sys/module/amdgpu") or os.path.isdir("/opt/rocm"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

# Ensure repo root is on sys.path so engine.py / track_video.py import cleanly
_REPO = Path(__file__).parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import gradio as gr

from track_video import track_video, summarize_tracks


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def run_tracker(
    video_path: str,
    confidence: float,
    tolerance: float,
    tracker: str,
    face_interval: int,
) -> tuple[str, str, str]:
    """
    Called by Gradio when user clicks Run.

    Returns:
        (annotated_video_path, json_download_path, summary_markdown)
    """
    if not video_path:
        return None, None, "No video uploaded."

    source = Path(video_path)

    # Write outputs to a temp directory that persists for this session
    out_dir = Path(tempfile.mkdtemp(prefix="faceid_track_"))
    annotated_path = out_dir / f"{source.stem}_annotated.mp4"
    json_path = out_dir / f"{source.stem}.tracks.json"

    try:
        frames = track_video(
            source=source,
            output_json=json_path,
            tracker=tracker,
            confidence=confidence,
            tolerance=tolerance,
            face_interval=face_interval,
            annotated_output=annotated_path,
            verbose=True,
        )
    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        return None, None, f"**Error during tracking:**\n\n```\n{err}\n```"

    summary = summarize_tracks(frames)

    # Build summary markdown table
    n_frames = len(frames)
    n_toby = sum(1 for r in summary if r["is_toby"])

    lines = [
        f"**Video:** `{source.name}`  ",
        f"**Frames processed:** {n_frames}  ",
        f"**Tracks found:** {len(summary)}  ",
        f"**Toby tracks:** {n_toby}  ",
        "",
        "| Track ID | Label | Frames Present | Avg Conf |",
        "|----------|-------|----------------|----------|",
    ]
    for row in summary:
        label = "**Toby**" if row["is_toby"] else "unknown"
        lines.append(
            f"| {row['track_id']} | {label} | {row['frames_present']} | {row['avg_conf']:.3f} |"
        )

    if n_toby == 0:
        lines += [
            "",
            "> **Note:** No Toby match found in this video. This is expected if the video",
            "> does not contain Toby. Upload a video with Toby to verify face-ID labelling.",
        ]

    summary_md = "\n".join(lines)

    # Return annotated video path (Gradio displays it), json path (for download),
    # and summary markdown
    return str(annotated_path), str(json_path), summary_md


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Face-ID Video Tracker — Use Case B v1") as demo:
        gr.Markdown(
            "## Face-ID Video Tracker\n"
            "**Use Case B v1 — tracking only. Safe-zone computation is fast-follow (v2).**\n\n"
            "All inference runs locally on your machine (GPU via ROCm for YOLO, CPU for dlib).\n"
            "No data leaves localhost."
        )

        with gr.Row():
            # Left column: inputs
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="Upload video",
                    sources=["upload"],
                )

                confidence_sl = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                    label="Detection confidence (YOLO)",
                    info="Lower = detect more people (more false positives). Default: 0.3",
                )
                tolerance_sl = gr.Slider(
                    minimum=0.30, maximum=0.70, value=0.50, step=0.01,
                    label="Face-match tolerance",
                    info="Calibrated at 0.50 (0% FAR vs 2500 LFW, 100% recall). Do not raise above 0.50.",
                )
                tracker_dd = gr.Dropdown(
                    choices=["bytetrack.yaml", "botsort.yaml"],
                    value="bytetrack.yaml",
                    label="Tracker algorithm",
                    info="ByteTrack: fast, stable IDs. BoTSORT: uses ReID, slower.",
                )
                face_interval_sl = gr.Slider(
                    minimum=5, maximum=120, value=30, step=5,
                    label="Face-ID sample interval (frames)",
                    info="How often to run face encoding per track. Higher = faster but less frequent ID updates.",
                )

                run_btn = gr.Button("Run tracker", variant="primary")

            # Right column: outputs
            with gr.Column(scale=2):
                annotated_video = gr.Video(
                    label="Annotated video (Toby=GREEN, others=GREY)",
                    interactive=False,
                )
                summary_md = gr.Markdown(
                    value="Upload a video and click **Run tracker** to start.",
                )
                json_download = gr.File(
                    label="Download per-frame JSON",
                    interactive=False,
                )

        run_btn.click(
            fn=run_tracker,
            inputs=[video_input, confidence_sl, tolerance_sl, tracker_dd, face_interval_sl],
            outputs=[annotated_video, json_download, summary_md],
        )

        gr.Markdown(
            "---\n"
            "**Per-frame JSON format:**\n"
            "```json\n"
            '[\n'
            '  {"frame": 0, "tracks": [\n'
            '    {"id": 1, "box_xywh_norm": [0.5, 0.3, 0.2, 0.6],\n'
            '     "is_toby": false, "conf": 0.87, "face_distance": null}\n'
            "  ]}\n"
            "]\n"
            "```\n"
            "`box_xywh_norm`: normalised [cx, cy, w, h] (0–1) — Remotion-consumable.\n"
            "`is_toby`: sticky label; true once confirmed below tolerance.\n"
            "`face_distance`: dlib Euclidean distance on frames where face-ID ran; null otherwise."
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    print(f"[app] torch: {torch.__version__}")
    print(f"[app] GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[app] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[app] WARNING: Running on CPU — YOLO will be slow")

    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",   # localhost only — no external access
        server_port=7860,
        share=False,               # never share=True (biometric data stays local)
        show_error=True,
    )
