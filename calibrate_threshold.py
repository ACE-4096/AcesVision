"""calibrate_threshold.py — Genuine vs impostor threshold calibration.

Downloads LFW deep-funneled images (~173 MB) to /tmp/lfw_calibration/,
encodes a large sample of impostor faces with the SAME pipeline as scan_photos.py
(YuNet detect + dlib ResNet encode), then computes genuine and impostor distance
distributions and reports FAR/recall at key thresholds.

Cleanup: removes /tmp/lfw_calibration/ on exit (unless --keep-downloads).

Usage:
    python calibrate_threshold.py [--max-impostors 2000] [--jobs 4] [--keep-downloads]
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
from PIL import Image, ImageOps

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

# -----------------------------------------------------------------------
# Step 1: Enroll Toby embeddings (same path as engine._load_known_encodings)
# -----------------------------------------------------------------------

def load_toby_encodings() -> list[np.ndarray]:
    """Return dlib 128-dim encodings for every enrolled image.

    DETECTOR ORDER: YuNet first, HOG fallback — MUST match engine._load_known_encodings
    and scan_photos._encode_image so the genuine LOO distances reflect the same embedding
    space as production queries. HOG-first here produced the invalidated calibration
    (ticket a3c3c709): genuine max was 0.418 but production genuine distances reach ~0.55
    due to the cross-detector ~0.13 penalty.
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
        # YuNet first — same order as scan_photos._encode_image and engine._load_known_encodings
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

    print(f"  Loaded {len(encs)} Toby encodings from {toby_dir}")
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

    # Collect one image per identity first (max diversity), then fill with more
    all_paths: list[Path] = []
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
# Step 4: Encode one impostor image (worker function — runs in subprocess)
# -----------------------------------------------------------------------

def _encode_impostor(path_str: str, yunet_path_str: str) -> list[float] | None:
    """
    Encode a single impostor image. Returns first face's 128-dim encoding
    as a list, or None if no face found.
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

def compute_genuine_distances(toby_encs: list[np.ndarray]) -> np.ndarray:
    """Leave-one-out: for each Toby encoding, min dist to ALL others."""
    E = np.stack(toby_encs)  # (N, 128)
    dists = []
    for i in range(len(E)):
        others = np.delete(E, i, axis=0)
        d = np.linalg.norm(others - E[i], axis=1)
        dists.append(float(d.min()))
    return np.array(dists)


def compute_impostor_distances(
    impostor_encs: list[np.ndarray],
    toby_encs: list[np.ndarray],
) -> np.ndarray:
    """For each impostor, min distance to ANY of Toby's enrolled encodings."""
    T = np.stack(toby_encs)  # (N_toby, 128)
    dists = []
    for imp in impostor_encs:
        d = np.linalg.norm(T - imp, axis=1)
        dists.append(float(d.min()))
    return np.array(dists)


# -----------------------------------------------------------------------
# Step 6: Report
# -----------------------------------------------------------------------

def percentiles_str(arr: np.ndarray) -> str:
    p = np.percentile(arr, [5, 10, 25, 50, 75, 90, 95])
    return (f"p5={p[0]:.3f} p10={p[1]:.3f} p25={p[2]:.3f} "
            f"p50={p[3]:.3f} p75={p[4]:.3f} p90={p[5]:.3f} p95={p[6]:.3f}")


def histogram_str(arr: np.ndarray, bins: int = 20, width: int = 40) -> str:
    counts, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))
    max_c = max(counts) if counts.max() > 0 else 1
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(width * c / max_c)
        lines.append(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}] {bar} {c}")
    return "\n".join(lines)


def report(
    genuine_dists: np.ndarray,
    impostor_dists: np.ndarray,
    thresholds: list[float],
) -> None:
    print("\n" + "=" * 70)
    print("THRESHOLD CALIBRATION REPORT")
    print("=" * 70)

    print(f"\nGENUINE distribution  (n={len(genuine_dists)} leave-one-out)")
    print(f"  min={genuine_dists.min():.3f}  max={genuine_dists.max():.3f}  "
          f"mean={genuine_dists.mean():.3f}  std={genuine_dists.std():.3f}")
    print(f"  {percentiles_str(genuine_dists)}")

    print(f"\nIMPOSTOR distribution  (n={len(impostor_dists)} LFW faces)")
    print(f"  min={impostor_dists.min():.3f}  max={impostor_dists.max():.3f}  "
          f"mean={impostor_dists.mean():.3f}  std={impostor_dists.std():.3f}")
    print(f"  {percentiles_str(impostor_dists)}")

    print("\nDISTRIBUTION HISTOGRAMS (distance 0.0 → 1.0)")
    print(f"\n  Genuine (n={len(genuine_dists)}):")
    print(histogram_str(genuine_dists))
    print(f"\n  Impostor (n={len(impostor_dists)}):")
    print(histogram_str(impostor_dists))

    print("\nTHRESHOLD ANALYSIS")
    print(f"  {'Threshold':>10}  {'FAR (%)':>10}  {'Recall (%)':>12}  "
          f"{'FA count':>10}  {'Miss count':>12}")
    print("  " + "-" * 60)

    recommended_thr = None
    for thr in sorted(thresholds):
        fa_count = int((impostor_dists <= thr).sum())
        far = 100.0 * fa_count / len(impostor_dists)
        recall_count = int((genuine_dists <= thr).sum())
        recall = 100.0 * recall_count / len(genuine_dists)
        miss_count = len(genuine_dists) - recall_count
        print(f"  {thr:>10.2f}  {far:>10.2f}  {recall:>12.2f}  "
              f"{fa_count:>10}  {miss_count:>12}")
        if far < 1.0 and recommended_thr is None:
            recommended_thr = thr

    # Sweep for exact <0.5% FAR threshold
    far_targets = [0.5, 1.0]
    print(f"\nFINE-GRAINED SWEEP (step 0.01)")
    print(f"  {'Threshold':>10}  {'FAR (%)':>10}  {'Recall (%)':>12}")
    print("  " + "-" * 40)
    prev_far = 100.0
    for thr_i in range(30, 70):
        thr = thr_i / 100.0
        fa = int((impostor_dists <= thr).sum())
        far = 100.0 * fa / len(impostor_dists)
        recall = 100.0 * (genuine_dists <= thr).sum() / len(genuine_dists)
        for ft in list(far_targets):
            if far <= ft and prev_far > ft:
                print(f"  *** FAR drops below {ft}% at thr={thr:.2f} ***")
                far_targets.remove(ft)
        print(f"  {thr:>10.2f}  {far:>10.3f}  {recall:>12.1f}")
        prev_far = far

    # Separation gap analysis
    g_max = genuine_dists.max()
    i_min = impostor_dists.min()
    overlap_g = (genuine_dists >= i_min).sum()
    overlap_i = (impostor_dists <= g_max).sum()

    print("\nSEPARATION VERDICT")
    if i_min > g_max:
        print(f"  CLEAN GAP: impostor min ({i_min:.3f}) > genuine max ({g_max:.3f})")
        print(f"  Perfect separation — any threshold in [{g_max:.3f}, {i_min:.3f}] is safe.")
    else:
        overlap_pct_i = 100.0 * overlap_i / len(impostor_dists)
        overlap_pct_g = 100.0 * overlap_g / len(genuine_dists)
        print(f"  OVERLAP: genuine_max={g_max:.3f}, impostor_min={i_min:.3f}")
        print(f"  {overlap_i} impostors ({overlap_pct_i:.1f}%) score <= genuine max")
        print(f"  {overlap_g} genuine ({overlap_pct_g:.1f}%) score >= impostor min")
        if overlap_pct_i < 2.0:
            print("  VERDICT: Tight overlap — usable at strict threshold (~<1% FAR achievable)")
        elif overlap_pct_i < 10.0:
            print("  VERDICT: Moderate overlap — usable with some recall trade-off")
        else:
            print("  VERDICT: HEAVY OVERLAP — distributions are NOT well-separated; "
                  "dlib ResNet may not be discriminative enough for this identity")

    if recommended_thr is not None:
        fa_final = int((impostor_dists <= recommended_thr).sum())
        far_final = 100.0 * fa_final / len(impostor_dists)
        rec_final = 100.0 * (genuine_dists <= recommended_thr).sum() / len(genuine_dists)
        print(f"\nRECOMMENDED THRESHOLD: {recommended_thr:.2f}")
        print(f"  FAR={far_final:.2f}%  Recall={rec_final:.1f}%")
    else:
        print("\nNo threshold in the tested range achieves FAR < 1% — see fine sweep above.")

    print("\n" + "=" * 70)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold calibration for face-id.")
    ap.add_argument("--max-impostors", type=int, default=2000,
                    help="Max LFW impostor images to encode (default: 2000)")
    ap.add_argument("--jobs", type=int,
                    default=max(1, (os.cpu_count() or 4) // 2),
                    help="Parallel encoding workers")
    ap.add_argument("--keep-downloads", action="store_true",
                    help="Do not delete /tmp/lfw_calibration after run")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
                    help="Threshold values to evaluate")
    args = ap.parse_args()

    print("=" * 70)
    print("STEP 1: Loading Toby's enrolled encodings")
    print("=" * 70)
    toby_encs = load_toby_encodings()
    if len(toby_encs) < 2:
        print("ERROR: Need at least 2 Toby encodings for leave-one-out. Exiting.")
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

    yunet_str = str(YUNET_PATH)
    impostor_encs: list[np.ndarray] = []
    failed = 0
    no_face = 0

    t0 = time.time()
    done = 0
    total = len(impostor_paths)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(_encode_impostor, str(p), yunet_str): p
            for p in impostor_paths
        }
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
                    impostor_encs.append(np.array(enc))
            except Exception as e:
                failed += 1

    print(f"\n  Encoded: {len(impostor_encs)}  no-face: {no_face}  errors: {failed}")

    if len(impostor_encs) < 100:
        print("ERROR: Too few impostor encodings. Check LFW download and detector.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("STEP 5: Computing genuine (leave-one-out) distances")
    print("=" * 70)
    genuine_dists = compute_genuine_distances(toby_encs)
    print(f"  {len(genuine_dists)} genuine distances computed")

    print("\n" + "=" * 70)
    print("STEP 6: Computing impostor distances (min dist to Toby gallery)")
    print("=" * 70)
    impostor_dists = compute_impostor_distances(impostor_encs, toby_encs)
    print(f"  {len(impostor_dists)} impostor distances computed")

    # Save raw numbers for reference
    np_out = LFW_TMP / "calibration_results.npz"
    np.savez(str(np_out), genuine=genuine_dists, impostor=impostor_dists)
    print(f"  Raw distances saved to {np_out}")

    report(genuine_dists, impostor_dists, args.thresholds)

    if not args.keep_downloads:
        print(f"\n[cleanup] Removing {LFW_TMP} ...")
        shutil.rmtree(str(LFW_TMP), ignore_errors=True)
        print("  Cleaned up.")
    else:
        print(f"\n[keep] LFW data retained at {LFW_TMP}")


if __name__ == "__main__":
    main()
