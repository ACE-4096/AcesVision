"""AcesVision local vision runtime.

``VisionPipeline`` is resolved lazily (PEP 562). Importing it eagerly pulled
OpenCV — and, through ``sources``, the camera layer — into every process that
touched any part of this package, including ``acesvision.catalog``, which is the
pure gesture vocabulary that ``gesture_catalog.py`` and any out-of-process
subscriber load. That module promises no cv2 and no device I/O, and a package
``__init__`` that imports the capture loop would quietly break the promise.

The public name is unchanged: ``from acesvision import VisionPipeline`` still
works and still returns the same class, it just costs the OpenCV import at the
moment you ask for it rather than at the moment you import a sibling.
"""

from .contracts import SceneFrame, SourceSpec

#: The emitter version advertised to subscribers in every event's
#: ``emitter.version`` field. Bump it when the wire behaviour changes; the event
#: ``schema`` is versioned separately and independently.
__version__ = "0.1.0"

__all__ = ["SceneFrame", "SourceSpec", "VisionPipeline", "__version__"]


def __getattr__(name):
    if name == "VisionPipeline":
        from .pipeline import VisionPipeline

        return VisionPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
