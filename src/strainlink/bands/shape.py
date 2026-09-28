"""Band shape classification: linear, log-spiral, or composite (log-spiral then straight).

Curves are written in intrinsic form: tangent angle theta as a function of
arc length s. A logarithmic spiral has curvature kappa(s) = kappa0 / (1 + k
|kappa0| s) (its radius of curvature grows linearly with s), which
integrates to theta(s) = alpha + sign(kappa0) * ln(1 + k |kappa0| s) / k.
Composite curves hold theta constant after a switch length s1.

Residuals are orthogonal: each data point to its nearest point on the
model curve. (Matching by cumulative arc length drifts, because noise
inflates the measured arc length.)

Models are fitted by least squares and compared with the Bayesian
Information Criterion (BIC). Neighbouring centreline points are strongly
correlated, so the BIC uses an effective sample size of one point per
band width rather than the raw point count; otherwise the more complex
model would win on noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize
from scipy.spatial import cKDTree

# Delta-BIC >= 6 is "strong" evidence (Kass & Raftery 1995); below it the label is ambiguous.
STRONG_EVIDENCE_BIC = 6.0
MIN_POINTS_FOR_CURVED = 8
# Integration step for the model curves (mm).
MODEL_STEP_MM = 0.5
# Parameter bounds for the log-spiral rate k = tan(phi); phi of about 3-72 degrees.
K_BOUNDS = (0.05, 3.0)
# Relative positions along the band tried as starting guesses for the composite switch point.
SWITCH_GUESSES = (0.25, 0.45, 0.65, 0.85)
# Resolution limits: a band cannot resolve a radius of curvature smaller than
# this many band widths, nor a spiral segment shorter than this many widths.
MIN_RADIUS_WIDTHS = 2.0
MIN_SEGMENT_WIDTHS = 2.0
# Residual scale (in band widths) beyond which the soft-L1 loss down-weights points.
ROBUST_SCALE_WIDTHS = 0.1
# How far a model may be extended backwards to reach the wall toe (band widths).
MAX_BACK_EXTENSION_WIDTHS = 4.0
# Nested-model degeneracy: a composite whose switch point is within this
# fraction of either end is really a log-spiral (or a line); a curved fit that
# turns by less than MIN_TURNING_DEG is really a line. Degenerate fits are
# the same hypothesis as the simpler model, so they are not counted as rivals.
DEGENERATE_SWITCH_FRACTION = 0.05
MIN_TURNING_DEG = 1.0
# Practical straightness: a band whose best curved model departs from its best
# straight line by less than this is called linear. Statistics can detect
# millimetre-scale curvature that no engineer would call a curved slip
# surface. The same definition labels the synthetic ground truth.
STRAIGHT_TOLERANCE_MM = 5.0
# Within +-this fraction of the tolerance the straight/curved call is flagged ambiguous.
STRAIGHT_BORDERLINE = 0.3


@dataclass
class ModelFit:
    kind: str
    params: np.ndarray
    rms_mm: float
    bic: float
    alpha_deg: float
    alpha_sigma_deg: float


@dataclass
class ShapeResult:
    label: str                 # "linear" | "log-spiral" | "composite"
    ambiguous: bool
    delta_bic: float           # BIC gap to the runner-up (larger = more certain)
    lift_off_deg: float        # tangent angle at the band start
    lift_off_sigma_deg: float
    fits: dict[str, ModelFit] = field(default_factory=dict)

    @property
    def best(self) -> ModelFit:
        return self.fits[self.label]


def _theta(kind: str, p: np.ndarray, s: np.ndarray) -> np.ndarray:
    if kind == "linear":
        return np.full_like(s, p[2])
    alpha, kappa0, k = p[2], p[3], p[4]
    s_eff = np.minimum(s, p[5]) if kind == "composite" else s
    # Negative s extends the spiral backwards; keep the log argument positive
    # (beyond the spiral's pole the curve is undefined, so it is held straight).
    argument = np.maximum(1.0 + k * abs(kappa0) * s_eff, 1e-6)
    return alpha + np.sign(kappa0) * np.log(argument) / k


def model_points(kind: str, p: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Model positions at arc lengths s, shape (n, 2). s < 0 extends the curve backwards."""
    s_min = min(float(np.min(s)), 0.0) if len(s) else 0.0
    s_max = max(float(np.max(s)), 0.0) if len(s) else 0.0
    grid = np.arange(s_min, s_max + MODEL_STEP_MM, MODEL_STEP_MM)
    theta = _theta(kind, p, grid)
    dx = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(theta[1:]) + np.cos(theta[:-1])) * MODEL_STEP_MM)])
    dz = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(theta[1:]) + np.sin(theta[:-1])) * MODEL_STEP_MM)])
    dx -= np.interp(0.0, grid, dx)
    dz -= np.interp(0.0, grid, dz)
    return np.column_stack([p[0] + np.interp(s, grid, dx), p[1] + np.interp(s, grid, dz)])


def _arc_length(points: np.ndarray) -> np.ndarray:
    seg = np.diff(points, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])


def _segment_angle(points: np.ndarray) -> float:
    d = points[-1] - points[0]
    return float(np.arctan2(d[1], d[0]))


# The model curve is generated this much longer than the data, so points near
# the far end always have a true nearest point rather than the curve's end.
MODEL_OVERSHOOT = 1.3


def _fit(kind: str, points: np.ndarray, s: np.ndarray, guess: np.ndarray, bounds,
         robust_scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Robust (soft-L1) fit: a few stray centreline points, e.g. a residual end
    hook, must not bend the whole model."""
    grid = np.arange(0.0, MODEL_OVERSHOOT * s[-1] + MODEL_STEP_MM, MODEL_STEP_MM)

    def residual(p: np.ndarray) -> np.ndarray:
        curve = model_points(kind, p, grid)
        _, idx = cKDTree(curve).query(points)
        return (points - curve[idx]).ravel()

    result = optimize.least_squares(residual, guess, bounds=bounds, x_scale="jac",
                                    loss="soft_l1", f_scale=robust_scale)
    plain = residual(result.x)
    return result.x, result.jac, float(np.sum(plain**2))


def _turning_deg(fit: ModelFit, length: float) -> float:
    theta = _theta(fit.kind, fit.params, np.array([0.0, length]))
    return float(np.degrees(abs(theta[1] - theta[0])))


def _reduces_to(fit: ModelFit, length: float) -> str:
    """The simplest model a fit is equivalent to, given the data length."""
    if fit.kind == "linear" or _turning_deg(fit, length) < MIN_TURNING_DEG:
        return "linear"
    if fit.kind == "composite":
        switch = fit.params[5] / length
        if switch >= 1.0 - DEGENERATE_SWITCH_FRACTION:
            return "log-spiral"
        if switch <= DEGENERATE_SWITCH_FRACTION:
            return "linear"
    return fit.kind


def lift_off_from_model(
    shape: "ShapeResult", toe: np.ndarray, span_mm: float, width_mm: float,
) -> tuple[float, float, float]:
    """Lift-off angle at the toe from the best-fit model: (angle, sigma, gap).

    Measured band starts usually fall short of the toe (the wall-friction zone
    blurs the band there), so the fitted model is extended backwards to its
    closest approach to the toe and the angle is taken over [toe, toe + span],
    the same definition as lift_off_angle. gap is the distance (mm) from the
    toe to the model's closest point, a check on the extrapolation.
    """
    angle, gap = _model_lift_off(shape.best, toe, span_mm, width_mm)
    # Model uncertainty: plausible alternative shapes (BIC within the "strong
    # evidence" gap of the winner) can imply different angles at the toe.
    # Their spread is added to the fit uncertainty, so an ambiguous shape
    # also makes the lift-off angle honestly uncertain.
    plausible = [f for f in shape.fits.values() if f.bic - shape.best.bic < STRONG_EVIDENCE_BIC]
    angles = [_model_lift_off(f, toe, span_mm, width_mm)[0] for f in plausible]
    spread = (max(angles) - min(angles)) / 2.0 if len(angles) > 1 else 0.0
    sigma = float(np.hypot(shape.best.alpha_sigma_deg, spread))
    return angle, sigma, gap


def _model_lift_off(fit: "ModelFit", toe: np.ndarray, span_mm: float, width_mm: float) -> tuple[float, float]:
    back = MAX_BACK_EXTENSION_WIDTHS * width_mm
    s = np.arange(-back, span_mm + MODEL_STEP_MM, MODEL_STEP_MM)
    curve = model_points(fit.kind, fit.params, s)
    nearest = int(np.argmin(np.hypot(*(curve - toe).T)))
    gap = float(np.hypot(*(curve[nearest] - toe)))
    s_toe = s[nearest]
    head = model_points(fit.kind, fit.params, np.arange(s_toe, s_toe + span_mm + MODEL_STEP_MM, MODEL_STEP_MM))
    angle, _ = lift_off_angle(head, span_mm, width_mm)
    return angle, gap


def lift_off_angle(points: np.ndarray, span_mm: float, width_mm: float) -> tuple[float, float]:
    """Mean direction (deg from horizontal) of the first span_mm of a band, with 1-sigma.

    A tangent at a single point cannot be resolved at finite band width and
    grid resolution, so the lift-off angle is defined as the direction of an
    orthogonal-regression line through the band's first span_mm. The same
    definition is applied to the ground truth.
    """
    s = _arc_length(points)
    head = points[s <= span_mm]
    if len(head) < 3:
        head = points[: min(3, len(points))]
    centre = head.mean(axis=0)
    _, singular, vt = np.linalg.svd(head - centre, full_matrices=False)
    direction = vt[0]
    travel = head[-1] - head[0]
    if np.dot(direction, travel) < 0:
        direction = -direction
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    # Angle uncertainty from the scatter about the line, with an effective
    # sample size of one point per band width (neighbours are correlated).
    n_eff = max(_arc_length(head)[-1] / max(width_mm, 1e-6), 2.0)
    spread = singular[0] / np.sqrt(len(head))
    scatter = singular[1] / np.sqrt(max(len(head) - 2, 1)) if len(singular) > 1 else 0.0
    sigma = float(np.degrees(scatter / (spread * np.sqrt(n_eff)))) if spread > 0 else np.nan
    return angle, sigma


def deviation_from_line(points: np.ndarray) -> float:
    """Largest distance (mm) of a polyline from its orthogonal-regression line."""
    centre = points.mean(axis=0)
    normal = np.linalg.svd(points - centre, full_matrices=False)[2][1]
    return float(np.max(np.abs((points - centre) @ normal)))


def classify(points: np.ndarray, band_width_mm: float,
             straight_tolerance_mm: float = STRAIGHT_TOLERANCE_MM,
             full_band: np.ndarray | None = None) -> ShapeResult:
    """Fit all models to an ordered centreline (from the band's start) and pick by BIC.

    full_band: the whole detected band when `points` is only its interior.
    Straightness is judged on the whole band (curvature grows with length).
    """
    points = np.asarray(points, dtype=float)
    s = _arc_length(points)
    length = s[-1]
    n_raw = len(points)
    n_eff = max(length / max(band_width_mm, 1e-6), 3.0)
    head = points[: max(n_raw // 6, 3)]
    tail = points[-max(n_raw // 6, 3):]
    alpha0 = _segment_angle(head)
    turning = _segment_angle(tail) - alpha0

    candidates: dict[str, list[np.ndarray]] = {
        "linear": [np.array([points[0, 0], points[0, 1], _segment_angle(points)])],
    }
    max_kappa = 1.0 / (MIN_RADIUS_WIDTHS * band_width_mm)
    min_segment = MIN_SEGMENT_WIDTHS * band_width_mm
    if n_raw >= MIN_POINTS_FOR_CURVED and length > 0:
        kappa = turning / length if abs(turning) > 1e-3 else 1e-4
        candidates["log-spiral"] = [np.array([points[0, 0], points[0, 1], alpha0, 2 * kappa, 0.8])]
        if length > 2 * min_segment:
            candidates["composite"] = [
                np.array([points[0, 0], points[0, 1], alpha0, 2 * kappa / f, 0.8, f * length])
                for f in SWITCH_GUESSES
            ]

    fits: dict[str, ModelFit] = {}
    for kind, guesses in candidates.items():
        best = None
        for guess in guesses:
            lower = [-np.inf, -np.inf, -np.pi, -max_kappa, K_BOUNDS[0]] + ([min_segment] if kind == "composite" else [])
            upper = [np.inf, np.inf, np.pi, max_kappa, K_BOUNDS[1]] + ([length] if kind == "composite" else [])
            n_par = len(guess)
            bounds = (lower[:n_par], upper[:n_par])
            guess = np.clip(guess, np.array(bounds[0]) + 1e-9, np.array(bounds[1]) - 1e-9)
            params, jac, rss = _fit(kind, points, s, guess, bounds, ROBUST_SCALE_WIDTHS * band_width_mm)
            if best is None or rss < best[2]:
                best = (params, jac, rss)
        params, jac, rss = best
        p = len(params)
        variance = rss / max(2 * n_raw - p, 1)
        bic = n_eff * np.log(max(rss / (2 * n_raw), 1e-12)) + p * np.log(n_eff)
        try:
            covariance = np.linalg.pinv(jac.T @ jac) * variance
            # Correlated points: inflate the naive uncertainty to the effective sample size.
            alpha_sigma = float(np.sqrt(max(covariance[2, 2], 0.0) * n_raw / n_eff))
        except np.linalg.LinAlgError:
            alpha_sigma = np.nan
        fits[kind] = ModelFit(kind=kind, params=params, rms_mm=float(np.sqrt(rss / n_raw)), bic=float(bic),
                              alpha_deg=float(np.degrees(params[2])),
                              alpha_sigma_deg=float(np.degrees(alpha_sigma)))

    ranked = sorted(fits.values(), key=lambda f: f.bic)
    winner = ranked[0]
    rivals = [f for f in ranked[1:] if _reduces_to(f, length) != winner.kind]
    delta = rivals[0].bic - winner.bic if rivals else np.inf
    ambiguous = bool(delta < STRONG_EVIDENCE_BIC)
    if winner.kind != "linear":
        departure = deviation_from_line(points if full_band is None else full_band)
        if departure < straight_tolerance_mm:
            winner, delta = fits["linear"], np.inf
            ambiguous = False
        borderline = abs(departure - straight_tolerance_mm) < STRAIGHT_BORDERLINE * straight_tolerance_mm
        ambiguous = ambiguous or borderline
    return ShapeResult(
        label=winner.kind, ambiguous=ambiguous, delta_bic=float(delta),
        lift_off_deg=winner.alpha_deg, lift_off_sigma_deg=winner.alpha_sigma_deg, fits=fits,
    )
