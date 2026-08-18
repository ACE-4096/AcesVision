#!/usr/bin/env bash
# Person-in-view alerter — one-line launcher.
#
#   ./watch.sh                         # watch DroidCam (192.168.1.187), both alerts
#   ./watch.sh --source 0             # local webcam index 0
#   WATCH_ALERT=desktop ./watch.sh    # desktop pop-ups only
#
# Runs the YOLO watcher in this repo's own venv — torch, ultralytics and dlib all
# live there. Pass-through args go to the script.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${FACE_ID_PYTHON:-$REPO/.venv/bin/python}"

if [[ ! -x "$PY" ]]; then
  echo "python not found at: $PY" >&2
  echo "set FACE_ID_PYTHON=/path/to/python (needs torch + ultralytics + dlib)" >&2
  exit 1
fi

if ! "$PY" -c 'import torch, ultralytics' 2>/dev/null; then
  echo "$PY cannot import torch + ultralytics." >&2
  echo "see README 'Runtime: one venv' for the ROCm install commands" >&2
  exit 1
fi

# gfx1030 is what an RX 6600 needs ROCm to pretend to be. Only set it when this
# host actually has the AMD stack loaded, and never over an explicit value.
if [[ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]] \
   && { [[ -d /sys/module/amdgpu ]] || [[ -d /opt/rocm ]]; }; then
  export HSA_OVERRIDE_GFX_VERSION=10.3.0
fi
export FACE_ID_PYTHON="$PY"

exec "$PY" "$REPO/watch_person.py" "$@"
