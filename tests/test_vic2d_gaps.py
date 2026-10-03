"""Gaps vs Vic-2D, closed: strain measures, field export, lens-distortion correction."""

import cv2
import numpy as np
import pytest

from strainlink.lens import LensModel, calibrate, distort
from strainlink.strain import compute_strain, measures


def test_strain_measures_exact_for_stretch_and_rotation():
    lam, angle = 1.1, np.radians(5.0)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    F = rot @ np.diag([lam, 1.0])
    x, z = np.meshgrid(np.arange(20.0), np.arange(20.0)[::-1])
    ux = (F[0, 0] - 1) * x + F[0, 1] * z
    uz = F[1, 0] * x + (F[1, 1] - 1) * z
    m = measures(compute_strain(ux, uz, 1.0, -1.0, 5))
    c = (10, 10)
    assert m["exx_eng"][c] == pytest.approx(lam - 1, abs=1e-9)
    assert m["rotation_deg"][c] == pytest.approx(5.0, abs=1e-9)
    almansi = 0.5 * (np.eye(2) - np.linalg.inv(F @ F.T))
    assert m["almansi_xx"][c] == pytest.approx(almansi[0, 0], abs=1e-9)
    assert m["almansi_xy"][c] == pytest.approx(almansi[0, 1], abs=1e-9)
    assert m["von_mises"][c] == pytest.approx(2 / np.sqrt(3) * np.log(lam), abs=1e-9)


def test_lens_fitted_from_a_checkerboard_and_corrects_positions():
    w, h, sq = 900, 600, 55
    k = np.array([[900.0, 0, (w - 1) / 2], [0, 900.0, (h - 1) / 2], [0, 0, 1]])
    true = LensModel(k, np.array([-0.25, 0.08, 0, 0, 0]), (w, h), 0, 0, (0, 0))
    board = np.full((h, w), 230.0)
    x0, y0 = (w - 14 * sq) // 2, (h - 9 * sq) // 2
    for i in range(14):
        for j in range(9):
            if (i + j) % 2 == 0:
                board[y0 + j * sq:y0 + (j + 1) * sq, x0 + i * sq:x0 + (i + 1) * sq] = 25.0
    seen = distort(cv2.GaussianBlur(board, (0, 0), 0.8), true)
    model = calibrate([seen], square_mm=5.0)
    assert model.board == (13, 8) and model.rms_px < 0.1
    assert model.mm_per_px == pytest.approx(5.0 / sq, rel=0.005)
    pts = np.stack(np.meshgrid(np.linspace(40, w - 40, 12), np.linspace(40, h - 40, 8)), -1).reshape(-1, 1, 2).astype(np.float32)
    fixed = cv2.undistortPoints(pts, model.camera_matrix, model.distortion, P=model.camera_matrix).reshape(-1, 2)
    exact = cv2.undistortPoints(pts, k, true.distortion, P=k).reshape(-1, 2)
    assert np.abs(fixed - exact).max() < 0.3          # the lens moved these points by up to ~35 px


def test_export_writes_every_quantity(tmp_path):
    from strainlink.export import export
    from strainlink.fields import QUANTITIES
    x, z = np.meshgrid(np.arange(10.0), np.arange(8.0)[::-1])
    ux = np.stack([0 * x, 0.01 * x]); uz = np.stack([0 * z, -0.003 * z])
    np.savez_compressed(tmp_path / "fields.npz", x_mm=x, z_mm=z, ux=ux, uz=uz, push_mm=np.array([0.0, 1.0]))
    out = export(tmp_path, "csv")
    import pandas as pd
    table = pd.read_csv(out / "photo_0002.csv")
    assert set(QUANTITIES) <= set(table.columns)
    assert table["exx_eng"].median() == pytest.approx(0.01, abs=1e-9)
    from scipy.io import loadmat
    mat = loadmat(export(tmp_path, "mat"))
    assert mat["von_mises"].shape == (2, 8, 10)
