"""scan_photos.py — batch face-search over a photo library.

Finds every image in --source that contains an enrolled person's face, using
the same detector and embedder as the live camera pipeline. Runs
headlessly; no cv2.imshow.

    --engine arcface (default)  YuNet + ArcFace ONNX. Scores COSINE
                                SIMILARITY: higher is better.
    --engine dlib               YuNet + dlib ResNet 128-d. Scores EUCLIDEAN
                                DISTANCE: lower is better.

The two scores are not interchangeable, which is why ``--tolerance`` (a dlib
distance) is refused for ArcFace and ``--threshold`` names the engine-agnostic
knob instead. The manifest records ``metric`` alongside every score for the
same reason.

Usage:
    python scan_photos.py --source /run/user/1000/gvfs/afc:.../DCIM
    python scan_photos.py --source ~/Pictures --jobs 8 --copy
    python scan_photos.py --source ~/Pictures --engine dlib --tolerance 0.55

Outputs:
    <output_dir>/manifest.json  — [{path, matched, best_score, metric, num_faces}]
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

import matching

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
# Encoding loader — delegate entirely to engine so the embedding space is
# IDENTICAL to what the live pipeline enrols with.
# ---------------------------------------------------------------------------

def load_enrolled(engine: str = "arcface", variant: str | None = None) -> tuple[list, list]:
    """Return (embeddings, names) for the requested engine.

    Never re-implements enrolment. ``engine.py`` owns it, this file asks.
    """
    if engine == "arcface":
        import engine as _engine
        encs, names, _pipeline = _engine.arcface_gallery(variant)
        return encs, names
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


# One ArcFace pipeline per worker process, built once by the pool initializer.
# An ONNX session is cheap to call and expensive to create.
_WORKER_PIPELINE = None


def _init_arcface_worker(variant: str) -> None:
    import arcface
    # Each worker already owns a core; letting ORT fan out inside it as well
    # just makes the workers fight each other for the same CPUs.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    global _WORKER_PIPELINE
    _WORKER_PIPELINE = arcface.ArcFacePipeline.load(variant)


def _encode_image_arcface(path_str: str) -> dict:
    """Embed every face in one image with ArcFace.

    Same return shape as _encode_image so the scan loop does not branch.
    """
    import arcface

    result = {"path": path_str, "encodings": [], "num_faces": 0, "error": None}
    image = arcface.load_bgr(path_str)
    if image is None:
        result["error"] = "could not open image"
        return result
    try:
        found = _WORKER_PIPELINE.detect(image)
    except Exception as exc:
        result["error"] = str(exc)
        return result
    result["num_faces"] = len(found)
    result["encodings"] = [[float(v) for v in f.embedding] for f in found]
    return result


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
    threshold,
) -> tuple[str | None, float]:
    """Return (name_or_None, best_score) in ``threshold``'s metric.

    ``threshold`` is a ``matching.Threshold``, never a float: the direction of
    "best" depends on the metric, and a bare number does not carry one.
    """
    return matching.match(enrolled_encs, enrolled_names, query_enc, threshold)[:2]


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

def _default_variant() -> str:
    import arcface
    return arcface.DEFAULT_VARIANT


def _is_better(score: float, incumbent: float, threshold) -> bool:
    """Direction-aware comparison. Never `<` on a similarity."""
    return (score > incumbent if threshold.higher_is_better
            else score < incumbent)

def scan(
    source: Path,
    output: Path,
    threshold,
    jobs: int,
    copy: bool,
    engine: str = "arcface",
    variant: str | None = None,
) -> None:
    t_start = time.time()

    # --- Load enrolled embeddings ---
    print(f"[load] Loading enrolled faces for engine={engine} ...")
    print(f"[load] {threshold.describe()}")
    enrolled_encs, enrolled_names = load_enrolled(engine, variant)
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
    print(f"[run] Encoding with {jobs} worker(s) ...")
    prog = _Progress(len(images))

    manifest: list[dict] = []
    n_matched = 0

    path_strs = [str(p) for p in images]

    if engine == "arcface":
        pool_kwargs = dict(max_workers=jobs, initializer=_init_arcface_worker,
                           initargs=(variant or _default_variant(),))

        def submit(pool, ps):
            return pool.submit(_encode_image_arcface, ps)
    else:
        yunet_str = str(YUNET_PATH)
        pool_kwargs = dict(max_workers=jobs)

        def submit(pool, ps):
            return pool.submit(_encode_image, ps, yunet_str)

    with concurrent.futures.ProcessPoolExecutor(**pool_kwargs) as pool:
        futures = [submit(pool, ps) for ps in path_strs]
        for future in concurrent.futures.as_completed(futures):
            prog.tick()
            res = future.result()
            path = Path(res["path"])

            if res["error"]:
                print(f"\n[skip] {path.name}: {res['error']}")
                manifest.append({
                    "path": str(path),
                    "matched": False,
                    "best_score": None,
                    "metric": threshold.metric,
                    "num_faces": 0,
                    "error": res["error"],
                })
                continue

            # Check each detected face against the enrolled set. "Best" runs in
            # the metric's own direction — max similarity, or min distance.
            matched = False
            best_score: float | None = None
            for enc_list in res["encodings"]:
                name, score = _best_match(enc_list, enrolled_encs,
                                          enrolled_names, threshold)
                if best_score is None or _is_better(score, best_score, threshold):
                    best_score = score
                if name is not None:
                    matched = True
                    break  # one confirmed face is enough

            manifest.append({
                "path": str(path),
                "matched": matched,
                "best_score": round(best_score, 4) if best_score is not None else None,
                "metric": threshold.metric,
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

  # 8 parallel workers, hard-copy matches instead of symlink
  python scan_photos.py --source ~/Pictures --jobs 8 --copy

  # The previous dlib pipeline, with its own calibrated distance
  python scan_photos.py --source ~/Pictures --engine dlib --tolerance 0.50

  # Auto-detect a GVFS-mounted phone
  python scan_photos.py
""",
    )
    parser.add_argument(
        "--engine",
        choices=["arcface", "dlib"],
        default="arcface",
        help="Embedder. arcface (default) scores cosine similarity; dlib "
             "scores Euclidean distance. Default: arcface",
    )
    parser.add_argument(
        "--arcface-model",
        default=None,
        help="ArcFace variant (w600k_r50 or w600k_mbf). Default: w600k_r50",
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
        default=None,
        help=(
            "dlib ONLY. dlib distance threshold — lower = stricter match. "
            "0.50 is calibrated against 2500 LFW impostors with the YuNet-first "
            "pipeline: FAR=0%%, Recall=100%% (genuine max=0.452, impostor min=0.500). "
            "Do not raise above 0.50 without re-calibrating — 0.60 accepts ~8%% of "
            "strangers. Refused with --engine arcface, whose scores run the "
            "other way. See calibrate_threshold.py."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override the calibrated threshold for the chosen engine, in that "
            "engine's own metric. An override has no measured FAR — it is a "
            "knob for experiments, not a number to ship."
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

    # --tolerance is a dlib Euclidean distance. Under ArcFace, "0.50" would
    # not be a stricter or looser version of the same thing — it would be a
    # floor on cosine similarity, a completely different decision. Refuse it
    # rather than silently reinterpret it.
    if args.tolerance is not None and args.engine != "dlib":
        parser.error(
            "--tolerance is a dlib Euclidean distance and means nothing to "
            f"--engine {args.engine}, which scores cosine similarity (higher "
            "is better). Use --threshold, or --engine dlib."
        )

    engine_key = "dlib" if args.engine == "dlib" else "arcface"
    override = args.threshold if args.threshold is not None else args.tolerance
    if override is None:
        threshold = matching.threshold_for(engine_key)
    else:
        threshold = matching.threshold_for(
            engine_key, env={matching.THRESHOLD_ENV[engine_key]: str(override)})

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

    scan(source, args.output, threshold, args.jobs, args.copy,
         engine=args.engine, variant=args.arcface_model)


if __name__ == "__main__":
    main()
