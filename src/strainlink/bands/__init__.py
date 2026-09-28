"""Automatic shear-band detection, labelling, shape classification and measurement."""

from .detect import DetectedBand, DetectionSettings, FrameBands, PrimaryMetrics, detect_bands
from .shape import ShapeResult, classify

__all__ = [
    "DetectedBand", "DetectionSettings", "FrameBands", "PrimaryMetrics", "ShapeResult",
    "classify", "detect_bands",
]
