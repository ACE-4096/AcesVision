"""scan_photos.py — batch face-search over a photo library.

Finds every image in --source that contains Toby's (or any enrolled person's)
face, using the same dlib ResNet 128-dim encoder and YuNet detector as the
live camera pipeline.  Runs headlessly; no cv2.imshow.

Usage:
    python scan_photos.py --source /run/user/1000/gvfs/afc:.../DCIM
    python scan_photos.py --source ~/Pictures --tolerance 0.55 --jobs 8 --copy

Outputs:
    <output_dir>/manifest.json  — [{path, matched, best_distance, num_faces}]
    <output_dir>/matches/       — symlinks (or copies with --copy) of matched images

Tickets resolved: 4c9d92f7 (batch scanner), fc99f189 (HEIC support)
Decision: bfc17d96
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# HEIC/HEIF support — patch Pillow globally so face_recognition.load_image_file
# transparently opens iPhone .heic files.  Must happen before any Pillow import.
# ---------------------------------------------------------------------------
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIC_ENABLED = True
except ImportError:
    _HEIC_ENABLED = False
    print("[warn] pillow-heif not installed — .heic/.heif files will be skipped")
    print("       Fix: pip install pillow-heif")

# Standard imports after HEIC registration
import cv2
import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Supported extensions (lower-case).  HEIC added when pillow-heif is present.
# ---------------------------------------------------------------------------
_JPEG_PNG = {".jpg", ".jpeg", ".png"}
_HEIC_EXT = {".heic", ".heif"} if _HEIC_ENABLED else set()
SUPPORTED_EXT = _JPEG_PNG | _HEIC_EXT

# ---------------------------------------------------------------------------
# Paths (mirror engine.py constants so _load_known_encodings works correctly)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent
KNOWN_DIR = _REPO / "known_faces"
YUNET_PATH = _REPO / "models" / "face_detection_yunet.onnx"


# ---------------------------------------------------------------------------
# Encoding loader — delegate entirely to engine._load_known_encodings so the
# embedding space is IDENTICAL to what was used during enroll.py.
# ---------------------------------------------------------------------------

def load_enrolled() -> tuple[list, list]:
    """Return (encodings, names) from engine._load_known_encodings."""
    import face_recognition
    from engine import _load_known_encodings
    return _load_known_encodings(face_recognition)


# ---------------------------------------------------------------------------
# Per-image worker — runs in a subprocess (ProcessPoolExecutor).
# Must not share state with main process; imports face_recognition locally.
# ---------------------------------------------------------------------------

def _open_image_rgb(path: Path) -> np.ndarray | None:
    """Open any supported image as RGB numpy array with EXIF rotation applied."""
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)  # honour EXIF orientation
            if img.mode != "RGB":
                img = img.convert("RGB")
            return np.asarray(img)
    except Exception as exc:
        print(f"[skip] {path.name}: {exc}", flush=True)
        return None


def _encode_image(
    path_str: str,
    yunet_path_str: str,
) -> dict:
    """
    Detect faces in one image and return raw encodings list.

    Returns a dict so it survives pickling across process boundary:
        {path, encodings: [[float,...], ...], num_faces: int, error: str|None}
    """
    import face_recognition
    import cv2 as _cv2
    import numpy as _np
    from PIL import Image as _Img, ImageOps as _Ops

    path = Path(path_str)
    result = {"path": path_str, "encodings": [], "num_faces": 0, "error": None}

    try:
        # Open with EXIF-corrected orientation
        with _Img.open(path) as img:
            img = _Ops.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = _np.asarray(img)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    # --- Detection: try YuNet first (angle-robust), fall back to HOG ---
    locations = []
    yn = None
    yunet_path = Path(yunet_path_str)
    if yunet_path.exists():
        try:
            bgr = _cv2.cvtColor(arr, _cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            yn = _cv2.FaceDetectorYN.create(
                str(yunet_path), "", (w, h), 0.5, 0.3, 5000
            )
            yn.setInputSize((w, h))
            _, faces = yn.detect(bgr)
            if faces is not None:
                for f in faces:
                    x, y, bw, bh = (int(v) for v in f[:4])
                    x, y = max(0, x), max(0, y)
                    locations.append((y, x + bw, y + bh, x))  # dlib order
        except Exception:
            locations = []

    if not locations:
        locations = face_recognition.face_locations(arr, model="hog")

    result["num_faces"] = len(locations)
    if not locations:
        return result

    # --- Encode ---
    encs = face_recognition.face_encodings(arr, locations)
    result["encodings"] = [e.tolist() for e in encs]
    return result


# ---------------------------------------------------------------------------
# Distance matching
# ---------------------------------------------------------------------------

def _best_match(
    query_enc: list | np.ndarray,
    enrolled_encs: list,
    enrolled_names: list,
    tolerance: float,
) -> tuple[str | None, float]:
    """Return (name_or_None, best_distance)."""
    if not enrolled_encs:
        return None, 1.0
    q = np.asarray(query_enc)
    dists = np.linalg.norm(np.asarray(enrolled_encs) - q, axis=1)
    bi = int(np.argmin(dists))
    dist = float(dists[bi])
    if dist <= tolerance:
        return enrolled_names[bi], dist
    return None, dist


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_images(source: Path) -> list[Path]:
    """Recursively find all supported image files under source."""
    found = []
    for p in source.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            found.append(p)
    return sorted(found)


# ---------------------------------------------------------------------------
# Auto-detect phone DCIM (best-effort, --source still takes priority)
# ---------------------------------------------------------------------------

def _autodetect_phone_dcim() -> Path | None:
    """
    Try to find a phone DCIM folder mounted via GVFS (iPhone AFC or Android MTP).
    Returns the path if found and non-empty, else None.
    """
    gvfs_base = Path(f"/run/user/{os.getuid()}/gvfs")
    if not gvfs_base.exists():
        return None
    for mount in gvfs_base.iterdir():
        # iPhone: afc:host=... or gphoto2:...; Android: mtp:host=...
        if not mount.is_dir():
            continue
        for candidate in [
            mount / "DCIM",
            mount / "Internal Storage" / "DCIM",
            mount / "Phone" / "DCIM",
        ]:
            if candidate.exists() and any(candidate.iterdir()):
                return candidate
    return None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _safe_link_or_copy(src: Path, dest_dir: Path, copy: bool) -> None:
    """Symlink (or copy) src into dest_dir, avoiding name collisions."""
    dest = dest_dir / src.name
    if dest.exists() or dest.is_symlink():
        # Disambiguate with parent folder name
        dest = dest_dir / f"{src.parent.name}__{src.name}"
    if dest.exists() or dest.is_symlink():
        dest = dest_dir / f"{hash(str(src)) & 0xFFFF:04x}__{src.name}"
    if copy:
        shutil.copy2(src, dest)
    else:
        dest.symlink_to(src.resolve())


# ---------------------------------------------------------------------------
# Progress printer (simple, no external dep)
# ---------------------------------------------------------------------------

class _Progress:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self._t0 = time.time()

    def tick(self, n: int = 1) -> None:
        self.done += n
        elapsed = time.time() - self._t0
        pct = 100 * self.done / max(self.total, 1)
        rate = self.done / max(elapsed, 0.001)
        eta = (self.total - self.done) / max(rate, 0.001)
        print(
            f"\r  {self.done}/{self.total}  {pct:.0f}%  "
            f"{rate:.1f} img/s  ETA {eta:.0f}s   ",
            end="",
            flush=True,
        )

    def done_line(self) -> None:
        print()  # newline after \r progress


# ---------------------------------------------------------------------------
# Main scan logic
# ---------------------------------------------------------------------------

def scan(
    source: Path,
    output: Path,
    tolerance: float,
    jobs: int,
    copy: bool,
) -> None:
    t_start = time.time()

    # --- Load enrolled embeddings ---
    print("[load] Loading enrolled face encodings...")
    enrolled_encs, enrolled_names = load_enrolled()
    if not enrolled_encs:
        print("[error] No enrolled faces found in known_faces/. Run enroll.py first.")
        sys.exit(1)
    people = sorted(set(enrolled_names))
    print(f"[load] {len(enrolled_encs)} encoding(s) — {len(people)} person(s): {people}")

    # --- Discover images ---
    print(f"[scan] Discovering images in {source} ...")
    images = discover_images(source)
    if not images:
        print(f"[error] No supported images found under {source}")
        sys.exit(1)
    print(f"[scan] Found {len(images)} image(s) "
          f"({sum(1 for p in images if p.suffix.lower() in _HEIC_EXT)} HEIC)")

    # --- Prepare output dirs ---
    matches_dir = output / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    # --- Parallel encoding ---
    print(f"[run] Encoding with {jobs} worker(s), tolerance={tolerance} ...")
    prog = _Progress(len(images))

    manifest: list[dict] = []
    n_matched = 0

    yunet_str = str(YUNET_PATH)
    path_strs = [str(p) for p in images]

    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_encode_image, ps, yunet_str): ps
            for ps in path_strs
        }
        for future in concurrent.futures.as_completed(futures):
            prog.tick()
            res = future.result()
            path = Path(res["path"])

            if res["error"]:
                print(f"\n[skip] {path.name}: {res['error']}")
                manifest.append({
                    "path": str(path),
                    "matched": False,
                    "best_distance": None,
                    "num_faces": 0,
                    "error": res["error"],
                })
                continue

            # Check each detected face against enrolled set
            matched = False
            best_dist: float | None = None
            for enc_list in res["encodings"]:
                name, dist = _best_match(enc_list, enrolled_encs, enrolled_names, tolerance)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                if name is not None:
                    matched = True
                    break  # one confirmed face is enough

            manifest.append({
                "path": str(path),
                "matched": matched,
                "best_distance": round(best_dist, 4) if best_dist is not None else None,
                "num_faces": res["num_faces"],
            })

            if matched:
                n_matched += 1
                _safe_link_or_copy(path, matches_dir, copy)

    prog.done_line()

    # --- Write manifest ---
    manifest_path = output / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t_start
    action = "copied" if copy else "symlinked"
    print(
        f"\n[done] Scanned {len(images)}, matched {n_matched} "
        f"({action} to {matches_dir})"
    )
    print(f"       Manifest: {manifest_path}")
    print(f"       Elapsed:  {elapsed:.1f}s  ({len(images)/elapsed:.1f} img/s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a photo folder for enrolled faces (CPU-only, no cloud).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan USB-mounted iPhone DCIM
  python scan_photos.py --source /run/user/1000/gvfs/afc:host=.../DCIM

  # Strict match, 8 parallel workers, hard-copy matches instead of symlink
  python scan_photos.py --source ~/Pictures --tolerance 0.50 --jobs 8 --copy

  # Auto-detect a GVFS-mounted phone
  python scan_photos.py
""",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Directory to scan (recurse). Omit to auto-detect a GVFS phone mount.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./photo_matches"),
        help="Output directory (default: ./photo_matches)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.50,
        help=(
            "dlib distance threshold — lower = stricter match. "
            "0.50 is calibrated against 2500 LFW impostors with the YuNet-first "
            "pipeline: FAR=0%%, Recall=100%% (genuine max=0.452, impostor min=0.500). "
            "Do not raise above 0.50 without re-calibrating — 0.60 accepts ~8%% of "
            "strangers. See calibrate_threshold.py. Default: 0.50"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel encoding workers (default: half of CPU count)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Hard-copy matched images to output/matches/ instead of symlinking",
    )
    args = parser.parse_args()

    # Resolve source
    source = args.source
    if source is None:
        print("[auto] No --source given, scanning for GVFS phone mount...")
        source = _autodetect_phone_dcim()
        if source is None:
            parser.error(
                "No phone DCIM found under GVFS. "
                "Mount your phone (see README § Mounting your phone) "
                "then re-run with --source <path>."
            )
        print(f"[auto] Found: {source}")

    if not source.exists():
        parser.error(f"--source does not exist: {source}")
    if not source.is_dir():
        parser.error(f"--source must be a directory: {source}")

    scan(source, args.output, args.tolerance, args.jobs, args.copy)


if __name__ == "__main__":
    main()
