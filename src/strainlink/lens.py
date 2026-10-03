"""Lens-distortion correction from a photo of a calibration grid (as Vic-2D does).

Real lenses bend straight lines (barrel or pincushion distortion). In 2D
DIC that shows as apparent strain growing towards the edges of the photo,
even when the specimen only translates. A photo of a checkerboard, in the
specimen's plane with the same camera, lens and focus, gives the lens
model: the camera matrix and radial (k1, k2) distortion, fitted with
OpenCV. Every photo is then corrected before DIC (resampled with a cubic
kernel), keeping the scale at the centre of the photo unchanged.

The board is found automatically (its number of inner corners is searched),
and when its square size is given it also yields the scale (mm per pixel).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Checkerboards searched for, as inner-corner counts (columns, rows), largest first.
BOARD_SIZES = sorted({(c, r) for c in range(4, 21) for r in range(3, 16) if c >= r}, key=lambda s: -s[0] * s[1])
# The board is searched on a copy reduced to at most this width (px), then refined at full size.
SEARCH_WIDTH_PX = 1200
# A fit worse than this (px RMS reprojection error) is flagged.
GOOD_FIT_PX = 0.5


@dataclass
class LensModel:
    camera_matrix: np.ndarray     # 3x3
    distortion: np.ndarray        # k1, k2, p1, p2, k3
    image_size: tuple[int, int]   # (width, height)
    rms_px: float                 # reprojection error of the fit
    corners: int
    board: tuple[int, int]
    mm_per_px: float | None = None

    def to_json(self) -> dict:
        return {"camera_matrix": self.camera_matrix.tolist(), "distortion": self.distortion.ravel().tolist(),
                "image_size": list(self.image_size), "rms_px": self.rms_px, "corners": self.corners,
                "board": list(self.board), "mm_per_px": self.mm_per_px}

    @classmethod
    def from_json(cls, d: dict) -> "LensModel":
        return cls(np.asarray(d["camera_matrix"], float), np.asarray(d["distortion"], float),
                   tuple(d["image_size"]), float(d["rms_px"]), int(d["corners"]), tuple(d["board"]), d.get("mm_per_px"))


def find_board(gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Inner corners (N, 2) of the checkerboard in a photo and its size, or None."""
    gray8 = cv2.normalize(np.asarray(gray, float), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    factor = min(1.0, SEARCH_WIDTH_PX / gray8.shape[1])
    small = cv2.resize(gray8, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else gray8
    for size in BOARD_SIZES:
        ok, corners = cv2.findChessboardCornersSB(small, size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = corners.reshape(-1, 2) / factor
            refined = cv2.cornerSubPix(gray8, corners.astype(np.float32).reshape(-1, 1, 2), (7, 7), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
            return refined.reshape(-1, 2), size
    return None


def calibrate(grid_photos: list[np.ndarray], square_mm: float | None = None) -> LensModel:
    """Lens model from one or more photos of a checkerboard (more photos, steadier fit)."""
    object_points, image_points, size = [], [], None
    for photo in grid_photos:
        found = find_board(photo)
        if found is None:
            continue
        corners, board = found
        if size is not None and board != size:
            continue
        size = board
        grid = np.zeros((board[0] * board[1], 3), np.float32)
        grid[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * (square_mm or 1.0)
        object_points.append(grid)
        image_points.append(corners.astype(np.float32).reshape(-1, 1, 2))
    if not image_points:
        raise ValueError("no checkerboard found in the calibration photo(s): the whole board must be visible, sharp and evenly lit")
    h, w = np.asarray(grid_photos[0]).shape[:2]
    # One flat view cannot separate everything: fix what it cannot determine (principal point at the
    # centre, square pixels, no tangential or k3 terms) and fit focal length with k1, k2.
    flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    k0 = np.array([[w, 0, (w - 1) / 2], [0, w, (h - 1) / 2], [0, 0, 1]], dtype=float)
    rms, k, dist, _, _ = cv2.calibrateCamera(object_points, image_points, (w, h), k0, None,
                                             flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS)
    mm_per_px = None
    if square_mm:
        # Scale in the corrected photo: mean corner spacing after undistortion.
        und = cv2.undistortPoints(image_points[0], k, dist, P=k).reshape(-1, 2)
        pts = und.reshape(size[1], size[0], 2)
        spacing = np.concatenate([np.hypot(*np.diff(pts, axis=1).reshape(-1, 2).T),
                                  np.hypot(*np.diff(pts, axis=0).reshape(-1, 2).T)])
        mm_per_px = float(square_mm / np.median(spacing))
    return LensModel(k, dist, (w, h), float(rms), sum(len(p) for p in image_points), size, mm_per_px)


class Undistorter:
    """Corrects photos for the lens (same size, same scale at the centre)."""

    def __init__(self, model: LensModel):
        w, h = model.image_size
        self.size = (w, h)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(model.camera_matrix, model.distortion, None,
                                                           model.camera_matrix, (w, h), cv2.CV_32FC1)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        if (image.shape[1], image.shape[0]) != self.size:
            raise ValueError(f"the photos ({image.shape[1]}×{image.shape[0]} px) and the calibration grid photo "
                             f"({self.size[0]}×{self.size[1]} px) must come from the same camera setting")
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(float)


class CorrectedImages:
    """A photo sequence corrected for the lens on loading (wraps any image list)."""

    def __init__(self, images, undistort: Undistorter):
        self.images, self.undistort = images, undistort
        self.shape = getattr(images, "shape", None)
        self.paths = getattr(images, "paths", None)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, k: int) -> np.ndarray:
        return self.undistort(self.images[k])


def distort(image: np.ndarray, model: LensModel) -> np.ndarray:
    """What the lens would record of an ideal (undistorted) image: for testing with known answers."""
    h, w = image.shape[:2]
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.stack([cols.ravel(), rows.ravel()], axis=1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(pts, model.camera_matrix, model.distortion, P=model.camera_matrix).reshape(h, w, 2)
    return cv2.remap(np.asarray(image, np.float32), ideal[..., 0], ideal[..., 1], cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT).astype(float)
