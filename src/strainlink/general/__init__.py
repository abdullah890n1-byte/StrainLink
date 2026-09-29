"""Any test, any object: DIC, strain, localisation zones and correlation without a retaining wall.

The retaining-wall pipeline (strainlink.pipeline) adds soil-mechanics
results specific to that test. This package handles everything else — a
tensile coupon, a Brazilian disc, a beam, a slope or a sandbox — with the
same validated engines: DIC, strain, lost-correlation refill, band
detection and tracking, and the correlation engine.
"""

from .pipeline import run

__all__ = ["run"]
