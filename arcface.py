"""ArcFace face embeddings on ONNX Runtime — CPU, and honestly so.

Why this exists
---------------
The face stage cost 77.6 ms at 640x480 and 65.9 ms of that — 85% — was the
dlib ResNet encoder. Detection was never the problem: YuNet finds a face in
about 4-12 ms. So the detector stays and only the *recogniser* is replaced.
Measured on this host, YuNet + ArcFace runs the whole stage in ~12 ms with
``w600k_mbf`` and ~34 ms with ``w600k_r50``.

YuNet already emits the five landmarks ArcFace alignment wants (``f[4:14]`` of
each detection row), in the order ArcFace's reference template expects. No
second detector is needed, and no second detection pass is paid for.

Everything here runs on the CPU
-------------------------------
``onnxruntime-rocm`` on this host advertises ``ROCMExecutionProvider`` from
``onnxruntime.get_available_providers()`` and then silently executes on the
CPU, because ``libhipblas.so.3`` and ``libamdhip64.so.7`` are not present
(the host is ROCm 6.3.1; the wheel wants 7.x). That is the same
"reports healthy, runs somewhere else" hazard ``acesvision/yolo_worker.py``
already documents for GPU devices.

Two rules follow, and both are enforced below:

1. ``get_available_providers()`` is never consulted to decide anything. It is
   a claim about the build, not about the session. Only
   ``InferenceSession.get_providers()`` — what the *live session* actually
   bound — is checked, by :func:`check_providers`.
2. The default provider list is CPU only. Nothing is asked for that cannot be
   verified, so there is nothing to be wrong about. Asking for a GPU provider
   is opt-in via ``FACE_ID_ARCFACE_PROVIDERS`` and fails loudly when the
   session does not bind it.

The honest limit of rule 1: a provider that binds successfully and *then*
falls back per-operator cannot be detected from Python at all. That is
precisely why the default asks for nothing but the CPU.

Thread safety
-------------
An ``InferenceSession`` is safe to call from several threads at once, so the
embedder deliberately holds no lock — the process-wide lock in ``engine.py``
exists for dlib and must not be extended to this path. Each *detector* still
gets its own ``cv2.FaceDetectorYN``, which is not shared.

Models
------
Weights are never downloaded automatically, matching
``acesvision/perception.py``. See the README for the fetch step.
"""
from __future__ import annotations

import os
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).parent / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet.onnx"

#: The two shipped recognisers. ``dim`` is the raw embedding width; both are
#: L2-normalised to the unit sphere before any comparison, which is what makes
#: a dot product a cosine similarity.
Variant = namedtuple("Variant", "name filename dim input_size")

VARIANTS = {
    "w600k_r50": Variant("w600k_r50", "w600k_r50.onnx", 512, 112),
    "w600k_mbf": Variant("w600k_mbf", "w600k_mbf.onnx", 512, 112),
}

DEFAULT_VARIANT = "w600k_r50"

#: ArcFace's canonical 112x112 destination landmarks, in image order:
#: subject's right eye, left eye, nose tip, right mouth corner, left mouth
#: corner. The subject's right eye appears on the *left* of the image, which
#: is why index 0 has the smaller x. YuNet emits its five points in this same
#: order, so ``f[4:14]`` maps onto this template with no re-ordering.
ARCFACE_REFERENCE_5PT = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)

#: One detected face: pixel box, the five landmarks, YuNet's own score, and
#: the L2-normalised embedding.
DetectedFace = namedtuple("DetectedFace", "x y w h landmarks score embedding")


class ProviderUnavailable(RuntimeError):
    """A requested ONNX execution provider was not bound by the session."""


class ModelMissing(FileNotFoundError):
    """A model file is not installed. Nothing is ever downloaded for you."""


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def umeyama(src, dst, estimate_scale=True):
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Umeyama (1991). Returns a 3x3 homogeneous matrix. This is the same
    estimator ``skimage.transform.SimilarityTransform`` uses and that
    InsightFace's own alignment depends on; it is reimplemented here so the
    repo does not grow a scikit-image dependency for eleven lines of algebra,
    and so the transform itself is directly testable.

    Unlike ``cv2.estimateAffinePartial2D`` this is deterministic and has no
    RANSAC threshold to tune — with five points and no outlier model, the
    closed-form least-squares fit is the right tool.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2:
        raise ValueError(f"point sets must match in shape: {src.shape} vs {dst.shape}")

    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    cov = dst_demean.T @ src_demean / num

    # d flips the last singular direction when the fit would otherwise be a
    # reflection. A mirrored face is not a valid alignment.
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(cov) < 0:
        d[dim - 1] = -1.0

    matrix = np.eye(dim + 1, dtype=np.float64)
    u, s, vt = np.linalg.svd(cov)
    rank = np.linalg.matrix_rank(cov)
    if rank == 0:
        raise ValueError("degenerate landmark set: all points coincide")
    if rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(vt) > 0:
            matrix[:dim, :dim] = u @ vt
        else:
            saved = d[dim - 1]
            d[dim - 1] = -1.0
            matrix[:dim, :dim] = u @ np.diag(d) @ vt
            d[dim - 1] = saved
    else:
        matrix[:dim, :dim] = u @ np.diag(d) @ vt

    scale = 1.0
    if estimate_scale:
        variance = src_demean.var(axis=0).sum()
        if variance <= 0:
            raise ValueError("degenerate landmark set: zero variance")
        scale = float((s @ d) / variance)

    matrix[:dim, dim] = dst_mean - scale * (matrix[:dim, :dim] @ src_mean)
    matrix[:dim, :dim] *= scale
    return matrix


def alignment_matrix(landmarks5, size=112):
    """The 2x3 affine that warps ``landmarks5`` onto the ArcFace template.

    ``landmarks5`` is (5, 2) in image pixels, YuNet order. ``size`` scales the
    112x112 reference; anything other than 112 is a research knob, the shipped
    models want 112.
    """
    pts = np.asarray(landmarks5, dtype=np.float64)
    if pts.shape != (5, 2):
        raise ValueError(f"expected 5 landmarks as (5, 2), got {pts.shape}")
    reference = ARCFACE_REFERENCE_5PT * (size / 112.0)
    return umeyama(pts, reference)[:2, :]


def align(image_bgr, landmarks5, size=112):
    """The 112x112 aligned crop ArcFace is trained to consume."""
    matrix = alignment_matrix(landmarks5, size=size)
    return cv2.warpAffine(image_bgr, matrix, (size, size), borderValue=0.0)


def yunet_landmarks(row):
    """The five (x, y) landmarks out of one YuNet detection row.

    A YuNet row is ``[x, y, w, h, *10 landmark coords, score]``. Slicing
    ``[4:14]`` is the whole contract, and it is the reason this pipeline needs
    no second detector.
    """
    row = np.asarray(row, dtype=np.float64)
    if row.shape[0] < 14:
        raise ValueError(f"YuNet row too short for landmarks: {row.shape}")
    return row[4:14].reshape(5, 2)


# ---------------------------------------------------------------------------
# Execution providers
# ---------------------------------------------------------------------------

def default_providers(env=None):
    """Providers to request, CPU only unless the operator says otherwise.

    Read at call time, not at import — the GUI imports this module early and
    freezing the configuration at import is how the YOLO worker got its own
    bug.
    """
    env = os.environ if env is None else env
    raw = env.get("FACE_ID_ARCFACE_PROVIDERS", "").strip()
    if not raw:
        return ["CPUExecutionProvider"]
    return [p.strip() for p in raw.split(",") if p.strip()]


def default_intra_op_threads(env=None, cpu_count=None):
    """How many threads one ONNX session may use inside a single operator.

    Left unset, onnxruntime sizes this pool to every core on the box, which is
    measurably the wrong answer here. On this 12-core host under normal load,
    one w600k_r50 embedding takes:

        1 thread   103.1 ms        4 threads    31.7 ms
        2 threads   55.2 ms        6 threads    27.6 ms
        3 threads   40.0 ms        default      48.5 ms

    The default loses to an explicit 4 because a 112x112 graph gives twelve
    threads too little work each to pay for the synchronisation, and because
    the session is shared by every camera thread — each of which is already a
    unit of parallelism. Oversubscribing turns extra cameras into contention
    rather than throughput.

    Half the cores, capped at six. ``FACE_ID_ARCFACE_THREADS`` overrides it,
    and 0 hands the decision back to onnxruntime.
    """
    env = os.environ if env is None else env
    raw = env.get("FACE_ID_ARCFACE_THREADS", "").strip()
    if raw:
        return max(0, int(raw))
    cores = cpu_count or os.cpu_count() or 4
    return max(1, min(6, cores // 2))


def check_providers(active, requested):
    """Raise unless every requested provider was actually bound.

    ``active`` must come from ``InferenceSession.get_providers()`` — the
    providers this *session* bound. It must never come from
    ``onnxruntime.get_available_providers()``, which on this host lists
    ``ROCMExecutionProvider`` for a build whose ROCm libraries are missing,
    and which therefore reports a capability that does not execute.

    Kept as a free function taking plain lists so the rule is testable without
    onnxruntime installed at all.
    """
    active = list(active)
    missing = [p for p in requested if p not in active]
    if missing:
        raise ProviderUnavailable(
            f"ONNX session did not bind {missing}; it bound {active}. "
            "The provider is advertised by the build but not usable on this "
            "host. Re-run without FACE_ID_ARCFACE_PROVIDERS to use the CPU, "
            "which is the supported configuration."
        )
    return active


def model_path(variant=DEFAULT_VARIANT, models_dir=None):
    """Where a variant's weights live. Existence is not checked here."""
    if variant not in VARIANTS:
        raise KeyError(f"unknown ArcFace variant {variant!r}; have {sorted(VARIANTS)}")
    base = MODELS_DIR if models_dir is None else Path(models_dir)
    return base / VARIANTS[variant].filename


# ---------------------------------------------------------------------------
# The embedder
# ---------------------------------------------------------------------------

class ArcFaceEmbedder:
    """One ONNX session plus the alignment it needs. No lock, by design."""

    def __init__(self, session, variant=DEFAULT_VARIANT):
        self.session = session
        self.variant = VARIANTS[variant] if isinstance(variant, str) else variant
        inputs = session.get_inputs()
        self.input_name = inputs[0].name
        self.output_name = session.get_outputs()[0].name
        #: What the session actually bound, not what the build advertises.
        self.providers = list(session.get_providers())

    @classmethod
    def load(cls, variant=DEFAULT_VARIANT, providers=None, models_dir=None):
        """Build a session for ``variant`` and verify its providers."""
        import onnxruntime as ort

        path = model_path(variant, models_dir=models_dir)
        if not path.is_file():
            raise ModelMissing(
                f"ArcFace model is not installed: {path}. "
                "AcesVision never downloads models automatically. "
                "See README 'Model binaries' for the fetch step."
            )
        requested = default_providers() if providers is None else list(providers)
        options = ort.SessionOptions()
        # One session shared across camera threads; let ORT use the box.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The InsightFace graphs declare a dynamic input ('None', 3, 112, 112)
        # but a *static* output shape of (1, 512). Any batch larger than one
        # therefore makes onnxruntime log
        #     VerifyOutputSizes: Expected shape from model of {1,512}
        #     does not match actual shape of {N,512}
        # once per call — on every multi-face frame, for ever. The annotation
        # is stale, not the arithmetic: a batched call returns bit-identical
        # rows to N serial calls, which the test suite asserts
        # (ArcFaceSessionTests.test_batched_matches_serial_exactly, max
        # absolute difference 0.0). Warnings are raised to error level to keep
        # that one known-false alarm out of the log; if this ever needs
        # revisiting, delete this line and the warning comes straight back.
        options.log_severity_level = 3
        threads = default_intra_op_threads()
        if threads:
            options.intra_op_num_threads = threads
        session = ort.InferenceSession(str(path), sess_options=options,
                                       providers=requested)
        # get_providers() — the session's own binding. Never
        # get_available_providers(); see this module's docstring.
        check_providers(session.get_providers(), requested)
        return cls(session, variant=variant)

    # -- embedding ---------------------------------------------------------

    def embed_aligned(self, aligned_bgr):
        """Embed one already-aligned 112x112 BGR crop. L2-normalised."""
        return self.embed_aligned_batch([aligned_bgr])[0]

    def embed_aligned_batch(self, aligned_bgrs):
        """Embed a batch of aligned crops. One session call, N rows out."""
        if not aligned_bgrs:
            return np.zeros((0, self.variant.dim), dtype=np.float32)
        size = self.variant.input_size
        blob = cv2.dnn.blobFromImages(
            list(aligned_bgrs),
            scalefactor=1.0 / 127.5,
            size=(size, size),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,   # the models are trained on RGB; frames here are BGR
        )
        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        return l2_normalise(np.asarray(out, dtype=np.float32))

    def embed(self, image_bgr, landmarks5):
        """Align by landmarks, then embed. The whole query path in one call."""
        return self.embed_aligned(align(image_bgr, landmarks5,
                                        size=self.variant.input_size))


def l2_normalise(matrix):
    """Rows onto the unit sphere, so a dot product is a cosine similarity."""
    array = np.atleast_2d(np.asarray(matrix, dtype=np.float32))
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    # A zero row would be a NaN factory; leave it at zero, it can only ever
    # score 0 similarity, which is the correct "no opinion".
    norms[norms == 0.0] = 1.0
    return array / norms


# ---------------------------------------------------------------------------
# The pipeline: YuNet detect -> align -> embed
# ---------------------------------------------------------------------------

class ArcFacePipeline:
    """YuNet detection and ArcFace embedding, as one object.

    This class is the *same-detector invariant* made structural. Enrolment
    (``engine``), the photo scanner (``scan_photos``) and the calibration run
    (``calibrate_threshold``) all embed through this one object, so the box
    and the landmarks that produced a gallery vector are produced by exactly
    the code that produces a query vector. The previous pipeline held that
    invariant in three hand-copied blocks with a comment asking future editors
    to keep them in step; when they drifted, genuine distances moved by 0.13
    and the calibration was void (ticket a3c3c709).

    Detections are ordered by YuNet score, descending. "The face" in a
    single-face path is therefore the most confident face, everywhere, rather
    than whichever one the detector happened to list first.
    """

    def __init__(self, embedder, detector=None, score_threshold=None,
                 nms_threshold=0.3, top_k=5000, yunet_path=None):
        self.embedder = embedder
        self.score_threshold = (
            float(os.environ.get("FACE_ID_YUNET_SCORE", "0.6"))
            if score_threshold is None else float(score_threshold)
        )
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.yunet_path = Path(yunet_path or YUNET_PATH)
        self._detector = detector if detector is not None else self._build_detector()

    def _build_detector(self):
        if not self.yunet_path.is_file():
            raise ModelMissing(
                f"YuNet model is not installed: {self.yunet_path}. "
                "AcesVision never downloads models automatically. "
                "See README 'Model binaries' for the fetch step."
            )
        return cv2.FaceDetectorYN.create(
            str(self.yunet_path), "", (320, 320),
            self.score_threshold, self.nms_threshold, self.top_k,
        )

    @classmethod
    def load(cls, variant=DEFAULT_VARIANT, providers=None, models_dir=None, **kwargs):
        return cls(ArcFaceEmbedder.load(variant, providers=providers,
                                        models_dir=models_dir), **kwargs)

    @property
    def variant(self):
        return self.embedder.variant

    def rows(self, image_bgr):
        """Raw YuNet rows for one BGR image, best score first."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        height, width = image_bgr.shape[:2]
        if height < 2 or width < 2:
            return []
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return []
        rows = [np.asarray(row, dtype=np.float64) for row in faces]
        rows.sort(key=lambda r: float(r[14]) if r.shape[0] > 14 else 0.0,
                  reverse=True)
        return rows

    def detect(self, image_bgr):
        """Every face in the frame, with an embedding each.

        One batched session call for the whole frame — a second face costs
        almost nothing on top of the first.
        """
        rows = self.rows(image_bgr)
        if not rows:
            return []
        size = self.embedder.variant.input_size
        crops = [align(image_bgr, yunet_landmarks(row), size=size) for row in rows]
        embeddings = self.embedder.embed_aligned_batch(crops)
        out = []
        for row, embedding in zip(rows, embeddings):
            x, y, w, h = (int(v) for v in row[:4])
            out.append(DetectedFace(
                max(0, x), max(0, y), w, h,
                yunet_landmarks(row),
                float(row[14]) if row.shape[0] > 14 else 0.0,
                embedding,
            ))
        return out

    def encode_best(self, image_bgr):
        """The embedding of the most confident face, or None.

        Enrolment, scanning and calibration all go through here. If this
        function is right, they cannot disagree.
        """
        found = self.detect(image_bgr)
        return found[0].embedding if found else None

    def encode_file(self, path):
        """``encode_best`` on a file, EXIF orientation honoured."""
        image = load_bgr(path)
        return None if image is None else self.encode_best(image)

    def embedding_space(self):
        """Identity of the embedding space these vectors live in.

        Gallery vectors are only comparable to query vectors from the same
        space. ``engine`` keys its enrolment cache on this string, which is
        what stops a stale gallery surviving an engine or variant switch.
        """
        return embedding_space_id(self.embedder.variant.name)


def embedding_space_id(variant=DEFAULT_VARIANT):
    return f"arcface:{variant}/yunet-5pt-align112"


def load_bgr(path):
    """Read an image file as BGR with EXIF orientation applied, or None.

    Pillow rather than ``cv2.imread`` because iPhone photos arrive rotated by
    metadata alone and HEIC needs the pillow-heif opener, which
    ``scan_photos`` registers.
    """
    from PIL import Image, ImageOps

    try:
        with Image.open(str(path)) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            rgb = np.asarray(image)
    except Exception:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
