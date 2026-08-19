"""Match metrics and per-engine thresholds.

The trap this module exists to close
------------------------------------
``0.50`` was a *dlib Euclidean distance*, calibrated against 2500 LFW
impostors (ticket a3c3c709): genuine max 0.452, impostor min 0.500, a clean
gap, FAR 0%. Lower was better and the number lived in one constant.

ArcFace scores *cosine similarity*. Different metric, opposite direction,
different scale. Reusing 0.50 would not be a slightly-wrong threshold, it
would be a threshold that accepts nearly every stranger on Earth: a cosine
similarity of 0.50 is a fairly *good* match, and ``score <= 0.50`` therefore
means "let in everyone who does not look like the enrolled person".

So a threshold here is not a float. It is a float bound to a metric, and any
comparison must present the metric its score was computed in. A Euclidean
score offered to a cosine threshold raises :class:`MetricMismatch` rather than
quietly returning an answer. Thresholds are stored per engine — there is no
shared constant to accidentally reuse.

Provenance is carried alongside the number. A threshold with no measured FAR
is not a threshold, it is a guess, and the type says which one you have.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

EUCLIDEAN_L2 = "euclidean_l2"
COSINE_SIMILARITY = "cosine_similarity"


class MetricMismatch(ValueError):
    """A score was compared against a threshold from a different metric."""


class NotCalibrated(RuntimeError):
    """An engine has no defensible threshold. It must not be used blind."""


@dataclass(frozen=True)
class Metric:
    """A scoring metric and, crucially, which direction means 'same person'."""

    name: str
    higher_is_better: bool
    #: Short suffix for anything a human reads, so a 0.62 cosine and a 0.62
    #: distance never look like the same number on screen.
    label: str
    description: str


METRICS = {
    EUCLIDEAN_L2: Metric(
        EUCLIDEAN_L2, False, "d",
        "L2 distance between dlib 128-d encodings; 0 is identical.",
    ),
    COSINE_SIMILARITY: Metric(
        COSINE_SIMILARITY, True, "cos",
        "Cosine similarity between L2-normalised ArcFace embeddings; "
        "1 is identical, 0 is unrelated.",
    ),
    # LBPH's own confidence is a chi-square distance on the histogram, not a
    # probability, and its scale is nothing like the other two. It is named
    # separately so it can never be compared to either.
    "lbph_chi_square": Metric(
        "lbph_chi_square", False, "chi2",
        "OpenCV LBPH prediction confidence; 0 is identical, ~70 is the "
        "usable limit. Not comparable to any embedding metric.",
    ),
}


@dataclass(frozen=True)
class Threshold:
    """A decision boundary, its metric, and the evidence for it."""

    engine: str
    metric: str
    value: float
    #: Measured false-accept rate at ``value``, in percent. ``None`` means
    #: nobody has measured it — see :meth:`require_evidence`.
    far_percent: Optional[float] = None
    recall_percent: Optional[float] = None
    n_impostors: Optional[int] = None
    #: Where the number came from: ticket, dataset, date.
    provenance: str = "uncalibrated"

    def __post_init__(self):
        if self.metric not in METRICS:
            raise KeyError(f"unknown metric {self.metric!r}")

    @property
    def higher_is_better(self) -> bool:
        return METRICS[self.metric].higher_is_better

    @property
    def label(self) -> str:
        return METRICS[self.metric].label

    def accepts(self, score, metric) -> bool:
        """Is ``score`` a match? ``metric`` is not optional on purpose.

        Every caller has to name the metric it computed in. That is the whole
        guard: it converts "wrong direction" from a silent 100%-FAR bug into
        an exception at the first comparison.
        """
        if metric != self.metric:
            raise MetricMismatch(
                f"{self.engine} threshold is {self.value} in {self.metric} "
                f"({'higher' if self.higher_is_better else 'lower'} is better) "
                f"but the score was computed in {metric!r}. These are not "
                "interchangeable and no conversion between them is defined."
            )
        return float(score) >= self.value if self.higher_is_better else float(score) <= self.value

    def require_evidence(self):
        """Refuse a threshold that has no measured false-accept rate."""
        if self.far_percent is None:
            raise NotCalibrated(
                f"{self.engine} has no measured FAR; run calibrate_threshold.py "
                f"--engine {self.engine} before relying on this number."
            )
        return self

    def describe(self) -> str:
        far = "FAR unmeasured" if self.far_percent is None else f"FAR {self.far_percent:.2f}%"
        return (f"{self.engine}: {self.metric} "
                f"{'>=' if self.higher_is_better else '<='} {self.value:g} "
                f"({far}, n={self.n_impostors}) [{self.provenance}]")


# ---------------------------------------------------------------------------
# Per-engine thresholds. One entry per engine — never one shared constant.
# ---------------------------------------------------------------------------

#: dlib's calibrated tolerance. Do not restore the old 0.60 default: it
#: accepted ~8.2% of strangers.
DLIB_TOLERANCE = Threshold(
    engine="dlib",
    metric=EUCLIDEAN_L2,
    value=0.50,
    far_percent=0.0,
    recall_percent=100.0,
    n_impostors=2500,
    provenance="ticket a3c3c709, 2026-06-06, LFW n=2500, clean gap [0.452, 0.500]",
)

#: ArcFace w600k_r50, measured by calibrate_threshold.py on 2026-08-19.
#:
#: Genuine: 66 enrolled photos, leave-one-out, min 0.7435 mean 0.8721.
#: Impostor: 2500 LFW deep-funneled faces across 2500 distinct identities,
#: best score against the enrolled gallery, max 0.2634 mean 0.0673.
#: The two distributions do not touch: a clean gap of 0.4801 wide.
#: 0.503 is the midpoint of that gap — the point furthest from both
#: distributions, so the least likely to be crossed by an unseen face on
#: either side.
#:
#: 0.503 is NOT the dlib 0.50 with a decimal added. The numbers are close by
#: coincidence and mean opposite things: this one is a FLOOR on cosine
#: similarity, dlib's is a CEILING on Euclidean distance. Threshold.accepts()
#: refuses to compare across the two, which is why the resemblance is a
#: readability hazard rather than a correctness one.
#:
#: What this number does NOT establish: the genuine side is 66 photos of one
#: person under enrolment conditions, so 100% recall is a statement about
#: that sample, not a general accuracy claim. Live camera frames score lower
#: than enrolment photos; if the enrolled person starts being missed, the
#: honest fix is to re-run the calibration against live-condition genuines,
#: not to nudge this number down.
ARCFACE_THRESHOLD = Threshold(
    engine="arcface",
    metric=COSINE_SIMILARITY,
    value=0.503,
    far_percent=0.0,
    recall_percent=100.0,
    n_impostors=2500,
    provenance=(
        "2026-08-19 calibrate_threshold.py --engine arcface --arcface-model "
        "w600k_r50; LFW deep-funneled n=2500 impostors vs 66 leave-one-out "
        "genuines; clean gap [0.2634, 0.7435]; midpoint; FAR 0.000% "
        "recall 100.00%"
    ),
)

LBPH_THRESHOLD = Threshold(
    engine="lbph",
    metric="lbph_chi_square",
    value=70.0,
    provenance="inherited default, never calibrated against an impostor set",
)

THRESHOLDS = {
    "arcface": ARCFACE_THRESHOLD,
    # The 'yunet' engine is YuNet detection with a dlib encoder: the metric and
    # the calibrated number are dlib's, so it shares dlib's entry by value.
    "yunet": DLIB_TOLERANCE,
    "dlib": DLIB_TOLERANCE,
    "lbph": LBPH_THRESHOLD,
}

#: Per-engine environment overrides. Deliberately *not* one variable: setting
#: FACE_ID_TOLERANCE=0.5 must not silently reconfigure ArcFace, because 0.5
#: means opposite things to the two engines.
THRESHOLD_ENV = {
    "arcface": "FACE_ID_ARCFACE_THRESHOLD",
    "yunet": "FACE_ID_TOLERANCE",
    "dlib": "FACE_ID_TOLERANCE",
    "lbph": "FACE_ID_LBPH_THRESH",
}


def threshold_for(engine, env=None) -> Threshold:
    """The threshold for ``engine``, honouring that engine's own override.

    Read from the environment at call time. An override keeps the engine's
    metric — you can move the boundary, you cannot change what it measures.
    """
    engine = str(engine).lower()
    if engine not in THRESHOLDS:
        raise KeyError(f"no threshold registered for engine {engine!r}; "
                       f"have {sorted(THRESHOLDS)}")
    base = THRESHOLDS[engine]
    env = os.environ if env is None else env
    raw = env.get(THRESHOLD_ENV[engine])
    if raw is None or str(raw).strip() == "":
        return base
    from dataclasses import replace
    return replace(
        base,
        value=float(raw),
        far_percent=None,
        recall_percent=None,
        n_impostors=None,
        provenance=f"{THRESHOLD_ENV[engine]}={raw} (operator override, unmeasured)",
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_gallery(gallery, query, metric):
    """Score one query vector against every gallery vector.

    Returns a 1-D array in ``metric``'s units. ArcFace embeddings are already
    L2-normalised by ``arcface.l2_normalise``; they are normalised again here
    because a gallery loaded from anywhere else has not promised that, and a
    dot product of unnormalised vectors is not a cosine of anything.
    """
    matrix = np.atleast_2d(np.asarray(gallery, dtype=np.float64))
    vector = np.asarray(query, dtype=np.float64).ravel()
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError(
            f"gallery is {matrix.shape[1]}-d but the query is {vector.shape[0]}-d; "
            "these embeddings are from different models and are not comparable"
        )
    if metric == EUCLIDEAN_L2:
        return np.linalg.norm(matrix - vector, axis=1)
    if metric == COSINE_SIMILARITY:
        gallery_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(vector)
        gallery_norms[gallery_norms == 0.0] = 1.0
        query_norm = query_norm or 1.0
        return (matrix @ vector) / (gallery_norms * query_norm)
    raise KeyError(f"no scorer for metric {metric!r}")


def best_index(scores, metric) -> int:
    """Index of the best score *in this metric's direction*."""
    scores = np.asarray(scores)
    return int(np.argmax(scores) if METRICS[metric].higher_is_better
               else np.argmin(scores))


def match(gallery, names, query, threshold):
    """Nearest enrolled vector -> ``(name_or_None, score, known)``.

    ``threshold`` must be a :class:`Threshold`. A bare float is refused: it is
    the exact shape of the bug this module exists to prevent, because a float
    carries neither its metric nor its direction.
    """
    if not isinstance(threshold, Threshold):
        raise TypeError(
            "match() needs a Threshold, not a bare "
            f"{type(threshold).__name__}. A float carries no metric, and "
            "0.50 means 'strict' to dlib and 'wide open' to ArcFace. Use "
            "matching.threshold_for(engine)."
        )
    if gallery is None or len(gallery) == 0:
        # No opinion is not a match, and the score must be the worst possible
        # value in this metric rather than a number that looks like a good one.
        worst = 0.0 if threshold.higher_is_better else float("inf")
        return None, worst, False
    scores = score_gallery(gallery, query, threshold.metric)
    index = best_index(scores, threshold.metric)
    score = float(scores[index])
    if threshold.accepts(score, threshold.metric):
        return names[index], score, True
    return None, score, False


def format_score(score, metric) -> str:
    """A score a human can read without having to remember the direction."""
    return f"{score:.2f} {METRICS[metric].label}"
