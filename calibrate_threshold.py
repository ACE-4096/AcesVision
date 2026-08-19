"""calibrate_threshold.py — Genuine vs impostor threshold calibration.

Downloads LFW deep-funneled images (~173 MB) to /tmp/lfw_calibration/, encodes
a large sample of impostor faces with the SAME pipeline the chosen engine uses
in production, then computes genuine and impostor score distributions and
reports FAR/recall at every threshold.

The output of this script is the only defensible source for a threshold. A
threshold with no measured false-accept rate is a guess, and ``matching.py``
refuses to ship one.

Engines
-------
``--engine arcface`` (default)
    YuNet detect + ArcFace ONNX embed, through ``arcface.ArcFacePipeline`` —
    the same object ``engine._build_arcface`` and ``scan_photos`` use, so the
    same-detector invariant holds by construction rather than by comment.
    Scores COSINE SIMILARITY: higher is better, and a threshold is a floor.

``--engine dlib``
    YuNet detect (HOG fallback) + dlib ResNet encode. Scores EUCLIDEAN
    DISTANCE: lower is better, and a threshold is a ceiling. This reproduces
    ticket a3c3c709.

The two numbers are not interchangeable in either direction. Every table
below is labelled with its metric for that reason.

Cleanup: removes /tmp/lfw_calibration/ on exit (unless --keep-downloads).

Usage:
    python calibrate_threshold.py --engine arcface --arcface-model w600k_r50
    python calibrate_threshold.py --engine arcface --arcface-model w600k_mbf
    python calibrate_threshold.py --engine dlib
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

# -----------------------------------------------------------------------
# HEIC registration (same as scan_photos.py — must be first)
# -----------------------------------------------------------------------
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

import cv2
import numpy as np

import arcface
import matching

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
_REPO = Path(__file__).parent
KNOWN_DIR = _REPO / "known_faces"
YUNET_PATH = _REPO / "models" / "face_detection_yunet.onnx"

LFW_TMP = Path("/tmp/lfw_calibration")
LFW_URL = "https://ndownloader.figshare.com/files/5976015"
LFW_TGZ = LFW_TMP / "lfw.tgz"
# Figshare file is lfw_funneled (13k images, 5749 identities) — valid for calibration
LFW_DIR = LFW_TMP / "lfw_funneled"

# Per-worker ArcFace pipeline, built once by the pool initializer. An ONNX
# session is cheap to call and expensive to create; one per process, not one
# per image.
_WORKER_PIPELINE = None


# -----------------------------------------------------------------------
# Step 1: Enrol genuine embeddings (the SAME path production enrols with)
# -----------------------------------------------------------------------

def load_genuine_arcface(variant: str) -> list[np.ndarray]:
    """ArcFace embeddings for every enrolled photo, via ArcFacePipeline.

    This is literally the function ``engine._build_arcface`` enrols with. If
    it is wrong, production is wrong the same way, which is the only kind of
    calibration worth having.
    """
    pipeline = arcface.ArcFacePipeline.load(variant)
    print(f"  ArcFace {variant} on providers {pipeline.embedder.providers}")
    print(f"  embedding space: {pipeline.embedding_space()}")

    encs: list[np.ndarray] = []
    toby_dir = KNOWN_DIR / "Toby"
    if not toby_dir.exists():
        raise RuntimeError(f"known_faces/Toby/ not found at {toby_dir}")
    for img_path in sorted(toby_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        emb = pipeline.encode_file(img_path)
        if emb is None:
            print(f"  [warn] no face in {img_path.name}, skipping")
            continue
        encs.append(emb)
    print(f"  Loaded {len(encs)} genuine embeddings from {toby_dir}")
    return encs


def load_genuine_dlib() -> list[np.ndarray]:
    """dlib 128-d encodings for every enrolled image.

    DETECTOR ORDER: YuNet first, HOG fallback — MUST match
    engine._load_known_encodings and scan_photos._encode_image so the genuine
    LOO distances reflect the same embedding space as production queries.
    HOG-first here produced the invalidated calibration (ticket a3c3c709):
    genuine max was 0.418 but production genuine distances reach ~0.55 due to
    the cross-detector ~0.13 penalty.
    """
    import face_recognition as _fr

    yn = None
    if YUNET_PATH.exists():
        yn = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.5, 0.3, 5000)

    IMG_EXT = {".jpg", ".jpeg", ".png"}
    encs: list[np.ndarray] = []
    toby_dir = KNOWN_DIR / "Toby"
    if not toby_dir.exists():
        raise RuntimeError(f"known_faces/Toby/ not found at {toby_dir}")

    for img_path in sorted(toby_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXT:
            continue
        arr = _fr.load_image_file(str(img_path))
        locs = []
        if yn is not None:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            yn.setInputSize((w, h))
            _, faces = yn.detect(bgr)
            if faces is not None:
                for f in faces:
                    x, y, bw, bh = (int(v) for v in f[:4])
                    x, y = max(0, x), max(0, y)
                    locs.append((y, x + bw, y + bh, x))
        if not locs:
            locs = _fr.face_locations(arr, model="hog")
        if not locs:
            print(f"  [warn] no face in {img_path.name}, skipping")
            continue
        found = _fr.face_encodings(arr, locs[:1])
        if found:
            encs.append(found[0])

    print(f"  Loaded {len(encs)} genuine encodings from {toby_dir}")
    return encs


# -----------------------------------------------------------------------
# Step 2: Download LFW deep-funneled
# -----------------------------------------------------------------------

def download_lfw() -> None:
    LFW_TMP.mkdir(parents=True, exist_ok=True)
    if LFW_DIR.exists() and any(LFW_DIR.iterdir()):
        print(f"  LFW already extracted at {LFW_DIR}")
        return

    if not LFW_TGZ.exists():
        print(f"  Downloading LFW deep-funneled from {LFW_URL} ...")
        print("  (first run: ~173 MB, subsequent runs skip this)")
        t0 = time.time()
        import requests as _req
        with _req.get(LFW_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(str(LFW_TGZ), "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = 100 * downloaded / max(total_size, 1)
                    mb = downloaded / 1_048_576
                    print(f"\r  {mb:.1f} MB / {total_size/1_048_576:.1f} MB  ({pct:.0f}%)",
                          end="", flush=True)
        print(f"\n  Download done in {time.time()-t0:.1f}s")
    else:
        print(f"  LFW archive already cached at {LFW_TGZ}")

    print(f"  Extracting to {LFW_TMP} ...")
    t0 = time.time()
    with tarfile.open(str(LFW_TGZ), "r:gz") as tf:
        tf.extractall(str(LFW_TMP))
    print(f"  Extracted in {time.time()-t0:.1f}s")


# -----------------------------------------------------------------------
# Step 3: Collect impostor image paths (exclude Toby by name)
# -----------------------------------------------------------------------

def collect_impostor_paths(max_impostors: int) -> list[Path]:
    """
    Return up to max_impostors image paths from LFW, excluding any person
    whose directory name contains 'Toby' or 'Bellramsay' (case-insensitive).
    Samples evenly across identities for diversity.
    """
    all_dirs = sorted(p for p in LFW_DIR.iterdir() if p.is_dir())
    # Exclude any accidental Toby entries in LFW (very unlikely but safe)
    impostor_dirs = [
        d for d in all_dirs
        if "toby" not in d.name.lower() and "bellramsay" not in d.name.lower()
    ]

    per_identity_first: list[Path] = []
    per_identity_extra: list[Path] = []

    for d in impostor_dirs:
        imgs = sorted(d.glob("*.jpg"))
        if not imgs:
            continue
        per_identity_first.append(imgs[0])
        per_identity_extra.extend(imgs[1:])

    # Prioritise breadth (one per identity) over depth
    rng = np.random.default_rng(42)
    rng.shuffle(per_identity_first)
    rng.shuffle(per_identity_extra)

    combined = per_identity_first + per_identity_extra
    selected = combined[:max_impostors]

    print(f"  LFW has {len(impostor_dirs)} identities; "
          f"selected {len(selected)} impostor images "
          f"({len(per_identity_first)} distinct identities covered)")
    return selected


# -----------------------------------------------------------------------
# Step 4: Encode one impostor image (worker functions — run in subprocesses)
# -----------------------------------------------------------------------

def _init_arcface_worker(variant: str) -> None:
    """One ArcFace pipeline per worker process, built once."""
    global _WORKER_PIPELINE
    # Each worker is already a whole core; letting ORT fan out inside it too
    # just makes the workers fight each other for the same CPUs.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER_PIPELINE = arcface.ArcFacePipeline.load(variant)


def _encode_impostor_arcface(path_str: str) -> list[float] | None:
    """Embed the most confident face in one impostor image, or None."""
    emb = _WORKER_PIPELINE.encode_file(Path(path_str))
    return None if emb is None else [float(v) for v in emb]


def _encode_impostor_dlib(path_str: str, yunet_path_str: str) -> list[float] | None:
    """
    Encode a single impostor image with the dlib pipeline. Returns the first
    face's 128-dim encoding as a list, or None if no face found.
    Runs inside a ProcessPoolExecutor — all imports must be local.
    """
    import face_recognition as _fr
    import cv2 as _cv2
    import numpy as _np
    from PIL import Image as _Img, ImageOps as _Ops

    path = Path(path_str)
    try:
        with _Img.open(path) as img:
            img = _Ops.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = _np.asarray(img)
    except Exception:
        return None

    # Detection: YuNet first (same as scan_photos.py), HOG fallback
    locations = []
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
                    locations.append((y, x + bw, y + bh, x))
        except Exception:
            locations = []

    if not locations:
        locations = _fr.face_locations(arr, model="hog")

    if not locations:
        return None

    encs = _fr.face_encodings(arr, locations[:1])
    if not encs:
        return None
    return encs[0].tolist()


# -----------------------------------------------------------------------
# Step 5: Compute distributions
# -----------------------------------------------------------------------

def compute_genuine_scores(genuine: list[np.ndarray], metric: str) -> np.ndarray:
    """Leave-one-out: each genuine vector's BEST score against the others.

    "Best" follows the metric's direction — the minimum distance, or the
    maximum similarity. Getting this backwards would silently invert the
    whole report, which is why the direction is read from METRICS and never
    hard-coded.
    """
    stacked = np.stack(genuine)
    scores = []
    for i in range(len(stacked)):
        others = np.delete(stacked, i, axis=0)
        row = matching.score_gallery(others, stacked[i], metric)
        scores.append(float(row[matching.best_index(row, metric)]))
    return np.array(scores)


def compute_impostor_scores(impostors: list[np.ndarray],
                            genuine: list[np.ndarray],
                            metric: str) -> np.ndarray:
    """Each impostor's BEST score against the enrolled gallery.

    Best, not average: a gallery is matched nearest-neighbour, so the impostor
    who gets in is the one whose single closest gallery photo lets them in.
    """
    gallery = np.stack(genuine)
    scores = []
    for imp in impostors:
        row = matching.score_gallery(gallery, imp, metric)
        scores.append(float(row[matching.best_index(row, metric)]))
    return np.array(scores)


def accepts(scores: np.ndarray, threshold: float, metric: str) -> np.ndarray:
    """Boolean mask of which scores this threshold would accept."""
    if matching.METRICS[metric].higher_is_better:
        return scores >= threshold
    return scores <= threshold


# -----------------------------------------------------------------------
# Step 6: Report
# -----------------------------------------------------------------------

def percentiles_str(arr: np.ndarray) -> str:
    p = np.percentile(arr, [5, 10, 25, 50, 75, 90, 95])
    return (f"p5={p[0]:.3f} p10={p[1]:.3f} p25={p[2]:.3f} "
            f"p50={p[3]:.3f} p75={p[4]:.3f} p90={p[5]:.3f} p95={p[6]:.3f}")


def histogram_str(arr: np.ndarray, lo: float, hi: float,
                  bins: int = 20, width: int = 40) -> str:
    counts, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    max_c = max(int(counts.max()), 1)
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(width * c / max_c)
        lines.append(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}] {bar} {c}")
    return "\n".join(lines)


def sweep_range(metric: str) -> np.ndarray:
    if metric == matching.COSINE_SIMILARITY:
        return np.round(np.arange(0.00, 1.001, 0.01), 3)
    return np.round(np.arange(0.30, 0.801, 0.01), 3)


def report(genuine_scores: np.ndarray, impostor_scores: np.ndarray,
           metric: str, engine_label: str) -> tuple[float | None, dict]:
    """Print the full report; return (recommended_threshold, stats)."""
    higher = matching.METRICS[metric].higher_is_better
    direction = "HIGHER is better" if higher else "LOWER is better"
    comparator = ">=" if higher else "<="

    print("\n" + "=" * 70)
    print(f"THRESHOLD CALIBRATION REPORT — {engine_label}")
    print(f"METRIC: {metric} ({direction}); a threshold accepts when "
          f"score {comparator} threshold")
    print("=" * 70)

    for title, arr in (("GENUINE  (leave-one-out)", genuine_scores),
                       ("IMPOSTOR (LFW faces)", impostor_scores)):
        print(f"\n{title}  n={len(arr)}")
        print(f"  min={arr.min():.4f}  max={arr.max():.4f}  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}")
        print(f"  {percentiles_str(arr)}")

    lo = min(0.0, float(min(genuine_scores.min(), impostor_scores.min())))
    hi = max(1.0, float(max(genuine_scores.max(), impostor_scores.max())))
    print(f"\nDISTRIBUTION HISTOGRAMS ({lo:.2f} -> {hi:.2f})")
    print(f"\n  Genuine (n={len(genuine_scores)}):")
    print(histogram_str(genuine_scores, lo, hi))
    print(f"\n  Impostor (n={len(impostor_scores)}):")
    print(histogram_str(impostor_scores, lo, hi))

    print(f"\nFULL SWEEP (step 0.01) — FAR = strangers accepted, "
          f"Recall = enrolled accepted")
    print(f"  {'Threshold':>10}  {'FAR (%)':>10}  {'Recall (%)':>12}  "
          f"{'FA count':>10}  {'Miss count':>12}")
    print("  " + "-" * 62)

    best_zero_far = None
    for thr in sweep_range(metric):
        fa = int(accepts(impostor_scores, thr, metric).sum())
        far = 100.0 * fa / len(impostor_scores)
        hit = int(accepts(genuine_scores, thr, metric).sum())
        recall = 100.0 * hit / len(genuine_scores)
        print(f"  {thr:>10.2f}  {far:>10.3f}  {recall:>12.2f}  "
              f"{fa:>10}  {len(genuine_scores) - hit:>12}")
        # The best threshold at FAR 0 is the most permissive one that still
        # lets nobody in: highest recall subject to zero false accepts.
        if far == 0.0 and (best_zero_far is None
                           or recall > best_zero_far[1]):
            best_zero_far = (float(thr), recall)

    # -- separation -------------------------------------------------------
    if higher:
        genuine_worst, impostor_best = genuine_scores.min(), impostor_scores.max()
        clean = impostor_best < genuine_worst
        gap = (impostor_best, genuine_worst)
        overlap_i = int((impostor_scores >= genuine_worst).sum())
        overlap_g = int((genuine_scores <= impostor_best).sum())
    else:
        genuine_worst, impostor_best = genuine_scores.max(), impostor_scores.min()
        clean = impostor_best > genuine_worst
        gap = (genuine_worst, impostor_best)
        overlap_i = int((impostor_scores <= genuine_worst).sum())
        overlap_g = int((genuine_scores >= impostor_best).sum())

    print("\nSEPARATION VERDICT")
    print(f"  worst genuine  = {genuine_worst:.4f}")
    print(f"  best impostor  = {impostor_best:.4f}")
    if clean:
        width = abs(gap[1] - gap[0])
        print(f"  CLEAN GAP of {width:.4f} — [{gap[0]:.4f}, {gap[1]:.4f}]")
        print("  Any threshold strictly inside the gap gives FAR 0% and "
              "recall 100% on this sample.")
    else:
        print(f"  OVERLAP: {overlap_i} impostors ({100.0*overlap_i/len(impostor_scores):.2f}%) "
              f"score at least as well as the worst genuine; "
              f"{overlap_g} genuine ({100.0*overlap_g/len(genuine_scores):.2f}%) "
              f"score no better than the best impostor.")
        print("  There is no threshold with both FAR 0% and recall 100%.")

    stats = {
        "metric": metric,
        "engine": engine_label,
        "n_genuine": int(len(genuine_scores)),
        "n_impostors": int(len(impostor_scores)),
        "genuine_worst": float(genuine_worst),
        "impostor_best": float(impostor_best),
        "clean_gap": bool(clean),
        "gap": [float(gap[0]), float(gap[1])],
    }

    recommended = None
    if clean:
        # Midpoint of the gap: maximum distance from both distributions, so
        # the smallest chance that an unseen face on either side crosses it.
        recommended = round(float((gap[0] + gap[1]) / 2.0), 3)
        fa = int(accepts(impostor_scores, recommended, metric).sum())
        hit = int(accepts(genuine_scores, recommended, metric).sum())
        print(f"\nRECOMMENDED THRESHOLD: {comparator} {recommended:.3f} "
              f"(midpoint of the clean gap)")
        print(f"  FAR = {100.0*fa/len(impostor_scores):.3f}%  "
              f"Recall = {100.0*hit/len(genuine_scores):.3f}%")
        stats["recommended"] = recommended
        stats["far_percent"] = 100.0 * fa / len(impostor_scores)
        stats["recall_percent"] = 100.0 * hit / len(genuine_scores)
    elif best_zero_far is not None:
        recommended = best_zero_far[0]
        print(f"\nRECOMMENDED THRESHOLD: {comparator} {recommended:.3f} "
              f"(most permissive threshold with zero false accepts)")
        print(f"  FAR = 0.000%  Recall = {best_zero_far[1]:.3f}%")
        stats["recommended"] = recommended
        stats["far_percent"] = 0.0
        stats["recall_percent"] = best_zero_far[1]
    else:
        print("\nNO THRESHOLD IN RANGE ACHIEVES FAR 0%. Do not ship a number "
              "from this run — read the sweep and choose an explicit "
              "FAR/recall trade-off, or improve the embedder.")

    print("\n" + "=" * 70)
    return recommended, stats


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold calibration for face-id.")
    ap.add_argument("--engine", choices=["arcface", "dlib"], default="arcface",
                    help="Which embedder to calibrate (default: arcface)")
    ap.add_argument("--arcface-model", default=arcface.DEFAULT_VARIANT,
                    choices=sorted(arcface.VARIANTS),
                    help=f"ArcFace variant (default: {arcface.DEFAULT_VARIANT})")
    ap.add_argument("--max-impostors", type=int, default=2500,
                    help="Max LFW impostor images to encode (default: 2500)")
    ap.add_argument("--jobs", type=int,
                    default=max(1, (os.cpu_count() or 4) // 2),
                    help="Parallel encoding workers")
    ap.add_argument("--keep-downloads", action="store_true",
                    help="Do not delete /tmp/lfw_calibration after run")
    args = ap.parse_args()

    if args.engine == "arcface":
        metric = matching.COSINE_SIMILARITY
        engine_label = f"arcface/{args.arcface_model}"
    else:
        metric = matching.EUCLIDEAN_L2
        engine_label = "dlib"

    print("=" * 70)
    print(f"STEP 1: Loading enrolled genuine embeddings ({engine_label})")
    print("=" * 70)
    genuine = (load_genuine_arcface(args.arcface_model) if args.engine == "arcface"
               else load_genuine_dlib())
    if len(genuine) < 2:
        print("ERROR: Need at least 2 genuine embeddings for leave-one-out. Exiting.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("STEP 2: Downloading LFW deep-funneled dataset")
    print("=" * 70)
    download_lfw()

    print("\n" + "=" * 70)
    print("STEP 3: Collecting impostor image paths")
    print("=" * 70)
    impostor_paths = collect_impostor_paths(args.max_impostors)

    print("\n" + "=" * 70)
    print(f"STEP 4: Encoding {len(impostor_paths)} impostor images "
          f"({args.jobs} workers)")
    print("=" * 70)

    impostor_encs: list[np.ndarray] = []
    failed = no_face = done = 0
    total = len(impostor_paths)
    t0 = time.time()

    if args.engine == "arcface":
        pool_kwargs = dict(max_workers=args.jobs,
                           initializer=_init_arcface_worker,
                           initargs=(args.arcface_model,))
        def submit(pool, path):
            return pool.submit(_encode_impostor_arcface, str(path))
    else:
        pool_kwargs = dict(max_workers=args.jobs)
        yunet_str = str(YUNET_PATH)
        def submit(pool, path):
            return pool.submit(_encode_impostor_dlib, str(path), yunet_str)

    with concurrent.futures.ProcessPoolExecutor(**pool_kwargs) as pool:
        futures = [submit(pool, p) for p in impostor_paths]
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 0.001)
                eta = (total - done) / max(rate, 0.001)
                print(f"\r  {done}/{total}  {rate:.1f} img/s  ETA {eta:.0f}s   ",
                      end="", flush=True)
            try:
                enc = fut.result()
                if enc is None:
                    no_face += 1
                else:
                    impostor_encs.append(np.array(enc, dtype=np.float64))
            except Exception as exc:
                failed += 1
                if failed <= 3:
                    print(f"\n  [error] {exc}")

    print(f"\n  Encoded: {len(impostor_encs)}  no-face: {no_face}  errors: {failed}")
    print(f"  Encode wall time: {time.time()-t0:.1f}s")

    if len(impostor_encs) < 100:
        print("ERROR: Too few impostor encodings. Check LFW download and detector.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("STEP 5: Computing genuine (leave-one-out) scores")
    print("=" * 70)
    genuine_scores = compute_genuine_scores(genuine, metric)
    print(f"  {len(genuine_scores)} genuine scores computed")

    print("\n" + "=" * 70)
    print("STEP 6: Computing impostor scores (best score vs enrolled gallery)")
    print("=" * 70)
    impostor_scores = compute_impostor_scores(impostor_encs, genuine, metric)
    print(f"  {len(impostor_scores)} impostor scores computed")

    out = LFW_TMP / f"calibration_{args.engine}_{args.arcface_model}.npz"
    np.savez(str(out), genuine=genuine_scores, impostor=impostor_scores,
             metric=metric, engine=engine_label)
    print(f"  Raw scores saved to {out}")

    recommended, _stats = report(genuine_scores, impostor_scores, metric, engine_label)

    if recommended is not None:
        print(f"\nTo adopt this, edit matching.{args.engine.upper()}_THRESHOLD "
              f"— value, far_percent, recall_percent, n_impostors and "
              f"provenance together. A value without the evidence beside it "
              f"is refused by Threshold.require_evidence().")

    if not args.keep_downloads:
        print(f"\n[cleanup] Removing {LFW_TMP} ...")
        shutil.rmtree(str(LFW_TMP), ignore_errors=True)
        print("  Cleaned up.")
    else:
        print(f"\n[keep] LFW data retained at {LFW_TMP}")


if __name__ == "__main__":
    main()
