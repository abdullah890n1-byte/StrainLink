"""Putting images and sensor logs on one timeline.

The camera and the data logger keep separate clocks, and neither records
the other. StrainLink links them through the one quantity both see: the
wall push. The logger records it directly (actuator displacement); DIC
sees it as the movement of the soil touching the wall, read at the wall
face from the soil movement next to it (see wall_push_from_dic). A scale
factor is still fitted with the offset: DIC push = scale * actuator
push(photo time + offset). Assuming scale = 1 biased the offset by 6 s
in testing.

Other modes: a known offset entered by the user, or photos triggered every
fixed push increment (as Vic-Snap can do).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize
from sklearn.isotonic import IsotonicRegression

OffsetSource = Literal["dic", "user", "push-per-frame"]
# Photos whose DIC wall push is below this (mm) are not used for the offset:
# the wall hasn't clearly started moving yet.
MIN_PUSH_FOR_SYNC_MM = 0.5
# Band of soil next to the wall used to measure the wall push from DIC (mm);
# widened to the nearest WALL_STRIP_COLUMNS columns of DIC points on coarse grids.
WALL_STRIP_MM = (2.0, 25.0)
WALL_STRIP_COLUMNS = 2
WALL_STRIP_HEIGHT_FRACTION = (0.2, 0.8)
# The wall's own movement is read where the soil touches it: a curve (quadratic) through the soil
# movement of the first columns of DIC points, evaluated at the wall face (x = 0). At least
# FACE_FIT_MIN_COLUMNS of them must be measured in a photo; otherwise the strip is used.
FACE_FIT_COLUMNS = 8
FACE_FIT_MIN_COLUMNS = 5


@dataclass
class Timeline:
    frame_logger_time_s: np.ndarray  # each photo's time on the logger clock
    frame_push_mm: np.ndarray        # wall push at each photo (from the actuator record)
    clock_offset_s: float            # logger time of the first photo
    offset_sigma_s: float
    source: OffsetSource
    soil_to_wall_ratio: float = 1.0  # DIC movement of the soil at the wall / actuator push
    push_sigma_mm: float = 0.0       # the clock uncertainty expressed as wall push

    def at_frames(self, logger_time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
        """A logger channel sampled at the photo times."""
        return np.interp(self.frame_logger_time_s, logger_time_s, values, left=np.nan, right=np.nan)


def wall_push_from_dic(ux_mm: np.ndarray, x_mm: np.ndarray, z_mm: np.ndarray, wall_height_mm: float,
                       valid: np.ndarray | None = None) -> np.ndarray:
    """Per-frame wall push seen by DIC: the horizontal movement of the soil where it touches the wall.

    The soil a few centimetres from the wall moves slightly less than the wall (it compresses,
    and later rides up the slip surface), and that lag grows through the test out of proportion
    to the push. Matching clocks on the soil in a strip next to the wall therefore placed every
    photo up to 0.06 mm of push late in loose sand. The lag vanishes at the wall face, so the
    movement is fitted against the distance from the wall and read at the face. Where too few
    columns are measured, the median of the strip next to the wall is used.

    ux_mm: (frames, N) horizontal displacement of tracked points; x_mm, z_mm: (N,) positions.
    """
    columns = np.unique(np.round(x_mm[np.isfinite(x_mm) & (x_mm >= WALL_STRIP_MM[0])], 6))
    far_edge = max(WALL_STRIP_MM[1], columns[min(WALL_STRIP_COLUMNS, len(columns)) - 1]) if len(columns) else WALL_STRIP_MM[1]
    strip = ((x_mm >= WALL_STRIP_MM[0]) & (x_mm <= far_edge)
             & (z_mm >= WALL_STRIP_HEIGHT_FRACTION[0] * wall_height_mm)
             & (z_mm <= WALL_STRIP_HEIGHT_FRACTION[1] * wall_height_mm))
    if not strip.any():
        raise ValueError("no DIC points next to the wall: check the wall-toe position and the region of interest")
    values = np.where(valid, ux_mm, np.nan) if valid is not None else ux_mm
    with np.errstate(all="ignore"):
        from_strip = np.nanmedian(values[:, strip], axis=1)
    rows = ((z_mm >= WALL_STRIP_HEIGHT_FRACTION[0] * wall_height_mm)
            & (z_mm <= WALL_STRIP_HEIGHT_FRACTION[1] * wall_height_mm) & np.isfinite(x_mm))
    near = columns[:FACE_FIT_COLUMNS]
    if len(near) < FACE_FIT_MIN_COLUMNS:
        return from_strip
    members = [rows & (np.abs(x_mm - c) < 1e-6) for c in near]
    push = from_strip.copy()
    for k in range(values.shape[0]):
        with np.errstate(all="ignore"):
            profile = np.array([np.nanmedian(values[k, m]) if m.any() else np.nan for m in members])
        known = np.isfinite(profile)
        if known.sum() >= FACE_FIT_MIN_COLUMNS:
            push[k] = np.polyfit(near[known], profile[known], 2)[2]        # the curve's value at x = 0
    return push


def monotone_push(logger_time_s: np.ndarray, push_mm: np.ndarray) -> np.ndarray:
    """Best-fit non-decreasing push record (isotonic regression).

    A running maximum would also be monotone, but on noisy data it is biased
    upward by about one noise level (1 s of clock offset in testing).
    """
    ok = np.isfinite(push_mm) & np.isfinite(logger_time_s)
    fitted = np.full_like(push_mm, np.nan, dtype=float)
    fitted[ok] = IsotonicRegression(increasing=True).fit_transform(logger_time_s[ok], push_mm[ok])
    return fitted


def _logger_time_at_push(logger_time_s: np.ndarray, push_mm: np.ndarray, target_mm: np.ndarray) -> np.ndarray:
    """Invert the (monotonic) push record: when had the actuator pushed by target_mm?"""
    order = monotone_push(logger_time_s, push_mm)
    rising = np.concatenate([[True], np.diff(order) > 0])
    return np.interp(target_mm, order[rising], logger_time_s[rising], left=np.nan, right=np.nan)


def sync_by_dic(frame_times_s: np.ndarray, dic_push_mm: np.ndarray,
                logger_time_s: np.ndarray, logger_push_mm: np.ndarray) -> Timeline:
    """Clock offset (and soil/wall movement ratio) from matching DIC wall push to the actuator record."""
    usable = np.isfinite(dic_push_mm) & (dic_push_mm > MIN_PUSH_FOR_SYNC_MM)
    usable &= dic_push_mm < np.nanmax(logger_push_mm)
    if usable.sum() < 3:
        raise ValueError("too few photos after the wall started moving to synchronise clocks")
    # Starting guess: per-photo offsets assuming the soil moves exactly with the wall.
    matched = _logger_time_at_push(logger_time_s, logger_push_mm, dic_push_mm[usable])
    start = float(np.nanmedian(matched - frame_times_s[usable]))
    order = monotone_push(logger_time_s, logger_push_mm)
    t_frames, d_frames = frame_times_s[usable], dic_push_mm[usable]

    offset, scale, sigma_stat = _fit_offset(t_frames, d_frames, logger_time_s, order, start)
    # The soil at the wall is not a perfect proxy for the wall: the relation
    # is slightly non-linear, so the purely statistical error was ~7x too
    # optimistic in testing. Two model-error terms are added: the disagreement
    # between fits to the early and late halves of the test, and the scatter
    # of the soil movement about the fitted model, converted to time through
    # the push rate. (In testing: true error 0.9 s, stated +-1.2 s, i.e.
    # about 0.01 mm of wall push.)
    half = len(t_frames) // 2
    sigma_halves = 0.0
    if half >= 3:
        early = _fit_offset(t_frames[:half], d_frames[:half], logger_time_s, order, offset)[0]
        late = _fit_offset(t_frames[half:], d_frames[half:], logger_time_s, order, offset)[0]
        sigma_halves = abs(early - late) / 2.0
    model = scale * np.interp(t_frames + offset, logger_time_s, order)
    rate = np.gradient(np.interp(t_frames + offset, logger_time_s, order), t_frames)
    rms_mm = float(np.sqrt(np.mean((d_frames - model) ** 2)))
    typical_rate = float(np.median(rate[rate > 0])) if (rate > 0).any() else np.inf
    sigma_scatter = rms_mm / typical_rate if np.isfinite(typical_rate) and typical_rate > 0 else 0.0
    sigma = float(np.sqrt(sigma_stat**2 + sigma_halves**2 + sigma_scatter**2))
    timeline = _timeline(frame_times_s, offset, sigma, "dic", logger_time_s, logger_push_mm)
    timeline.soil_to_wall_ratio = scale
    timeline.push_sigma_mm = sigma * typical_rate if np.isfinite(typical_rate) else float("nan")
    return timeline


def _fit_offset(t_frames: np.ndarray, d_frames: np.ndarray, logger_time_s: np.ndarray, push_order: np.ndarray,
                start: float) -> tuple[float, float, float]:
    """Least-squares (offset, scale) with the offset's statistical 1-sigma."""
    def residual(p: np.ndarray) -> np.ndarray:
        offset, scale = p
        return d_frames - scale * np.interp(t_frames + offset, logger_time_s, push_order)

    fit = optimize.least_squares(residual, [start, 1.0], loss="soft_l1", f_scale=0.05)
    dof = max(len(d_frames) - 2, 1)
    variance = float(np.sum(residual(fit.x) ** 2) / dof)
    try:
        covariance = np.linalg.inv(fit.jac.T @ fit.jac) * variance
        sigma = float(np.sqrt(max(covariance[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        sigma = float("nan")
    return float(fit.x[0]), float(fit.x[1]), sigma


def sync_by_offset(frame_times_s: np.ndarray, offset_s: float,
                   logger_time_s: np.ndarray, logger_push_mm: np.ndarray) -> Timeline:
    return _timeline(frame_times_s, offset_s, 0.0, "user", logger_time_s, logger_push_mm)


def sync_by_push_step(n_frames: int, push_step_mm: float,
                      logger_time_s: np.ndarray, logger_push_mm: np.ndarray) -> Timeline:
    """Photos taken every push_step_mm of wall push (first photo at zero push)."""
    push = np.arange(n_frames) * push_step_mm
    times = _logger_time_at_push(logger_time_s, logger_push_mm, np.maximum(push, 1e-6))
    times[0] = logger_time_s[np.argmax(logger_push_mm > 1e-3)] if (logger_push_mm > 1e-3).any() else 0.0
    return Timeline(frame_logger_time_s=times, frame_push_mm=push, clock_offset_s=float(times[0]),
                    offset_sigma_s=0.0, source="push-per-frame")


def _timeline(frame_times_s, offset, sigma, source, logger_time_s, logger_push_mm) -> Timeline:
    logger_times = frame_times_s + offset
    push = np.interp(logger_times, logger_time_s, logger_push_mm)
    return Timeline(frame_logger_time_s=logger_times, frame_push_mm=np.clip(push, 0.0, None),
                    clock_offset_s=offset, offset_sigma_s=sigma, source=source)
