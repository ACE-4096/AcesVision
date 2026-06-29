#!/usr/bin/env bash
# Person-in-view alerter — one-line launcher.
#
#   ./watch.sh                         # watch DroidCam (192.168.1.187), both alerts
#   ./watch.sh --source 0             # local webcam index 0
#   WATCH_ALERT=desktop ./watch.sh    # desktop pop-ups only
#
# Runs the YOLO watcher under the torch-capable cv-worker venv and lets it shell
# out to the face-id venv for face recognition. Pass-through args go to the script.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CV_PY="${CV_WORKER_PYTHON:-$HOME/Documents/git_private/cv-worker/.venv/bin/python}"

if [[ ! -x "$CV_PY" ]]; then
  echo "torch-capable python not found at: $CV_PY" >&2
  echo "set CV_WORKER_PYTHON=/path/to/python (needs torch + ultralytics)" >&2
  exit 1
fi

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"  # AMD ROCm
export FACE_ID_PYTHON="${FACE_ID_PYTHON:-$REPO/.venv/bin/python}"      # dlib recogniser

exec "$CV_PY" "$REPO/watch_person.py" "$@"
