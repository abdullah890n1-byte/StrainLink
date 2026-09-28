"""Strain computation against analytical deformation fields."""

import numpy as np
import pytest

from strainlink.strain import compute_strain, displacement_gradients

SPACING = 2.0


def _grid(n=41):
    coords = np.arange(n) * SPACING
    return np.meshgrid(coords, coords)


def test_gradients_of_linear_field_are_exact():
    x, y = _grid()
    field = 0.003 * x - 0.002 * y + 1.0
    dfdx, dfdy = displacement_gradients(field, SPACING, SPACING, window=5)
    assert np.nanmax(np.abs(dfdx - 0.003)) < 1e-12
    assert np.nanmax(np.abs(dfdy + 0.002)) < 1e-12


def test_gradients_tolerate_holes():
    x, y = _grid()
    field = 0.01 * x
    field[10:13, 10:13] = np.nan
    dfdx, _ = displacement_gradients(field, SPACING, SPACING, window=5)
    finite = np.isfinite(dfdx)
    assert finite.sum() > 0.9 * field.size - 9
    assert np.nanmax(np.abs(dfdx[finite] - 0.01)) < 1e-12


def test_negative_row_spacing_flips_sign():
    x, y = _grid()
    field = 0.01 * y
    _, up = displacement_gradients(field, SPACING, -SPACING, window=5)
    assert np.nanmedian(up) == pytest.approx(-0.01)


def test_simple_shear_max_shear_strain():
    """Simple shear u = g*y: Hencky principal strains are +-asinh(g/2)."""
    x, y = _grid()
    g = 0.2
    strain = compute_strain(g * y, np.zeros_like(y), SPACING, SPACING, window=5)
    expected = 2 * np.arcsinh(g / 2)
    assert np.nanmedian(strain.gamma_max) == pytest.approx(expected, rel=1e-9)
    assert np.nanmedian(strain.volumetric) == pytest.approx(0.0, abs=1e-12)


def test_uniaxial_stretch_green_lagrange_and_hencky():
    x, y = _grid()
    stretch = 1.05
    strain = compute_strain((stretch - 1) * x, np.zeros_like(y), SPACING, SPACING, window=5)
    assert np.nanmedian(strain.exx) == pytest.approx(0.5 * (stretch**2 - 1))
    assert np.nanmedian(strain.e1) == pytest.approx(np.log(stretch))
    assert np.nanmedian(strain.volumetric) == pytest.approx(np.log(stretch))
