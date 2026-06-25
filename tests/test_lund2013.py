"""Tests for the Lund2013 real-data loader."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io

from synthetic_eye_tracking.detection import (
    BLINK,
    FIXATION,
    LABELS,
    OTHER,
    SACCADE,
    pix_to_deg,
    validate_samples,
)
from synthetic_eye_tracking.detection.lund2013 import (
    download_lund2013,
    load_lund2013_samples,
)

_SCREEN_RES = [1024.0, 768.0]
_SCREEN_DIM = [0.38, 0.30]
_VIEW_DIST = 0.67
_SAMP_FREQ = 500


def _make_pos() -> np.ndarray:
    """Build a small pos array with a couple of each integer label."""
    # columns: [ts_us, x_deg, y_deg, x_pix, y_pix, label]
    rows = [
        # ts (us),     x_deg, y_deg, x_pix, y_pix, label
        [1_000_000.0, 0.0, 0.0, 512.0, 384.0, 1],  # fixation
        [1_002_000.0, 0.0, 0.0, 520.0, 380.0, 1],  # fixation
        [1_004_000.0, 0.0, 0.0, 700.0, 300.0, 2],  # saccade
        [1_006_000.0, 0.0, 0.0, 800.0, 250.0, 2],  # saccade
        [1_008_000.0, 0.0, 0.0, 600.0, 400.0, 3],  # PSO -> other
        [1_010_000.0, 0.0, 0.0, 610.0, 410.0, 4],  # pursuit -> other
        [1_012_000.0, 0.0, 0.0, np.nan, np.nan, 5],  # blink (NaN pos)
        [1_014_000.0, 0.0, 0.0, np.nan, np.nan, 5],  # blink (NaN pos)
        [1_016_000.0, 0.0, 0.0, 512.0, 384.0, 6],  # undefined -> other
    ]
    return np.array(rows, dtype=float)


def _write_fake_mat(path) -> np.ndarray:
    pos = _make_pos()
    scipy.io.savemat(
        str(path),
        {
            "ETdata": {
                "pos": pos,
                "sampFreq": _SAMP_FREQ,
                "screenDim": _SCREEN_DIM,
                "screenRes": _SCREEN_RES,
                "viewDist": _VIEW_DIST,
            }
        },
    )
    return pos


def test_load_fake_tree(tmp_path):
    """Network-free: load a synthetic dataset tree and validate the frame."""
    annotated = tmp_path / "annotated_data" / "data used in the article"
    img_dir = annotated / "img"
    dots_dir = annotated / "dots"
    img_dir.mkdir(parents=True)
    dots_dir.mkdir(parents=True)

    pos = _write_fake_mat(img_dir / "S1_img_test_labelled_RA.mat")
    _write_fake_mat(dots_dir / "S2_dots_test_labelled_RA.mat")
    # An MN file that must be ignored when annotator="RA".
    _write_fake_mat(img_dir / "S1_img_test_labelled_MN.mat")

    df = load_lund2013_samples(
        root=tmp_path, annotator="RA", categories=("img", "dots", "video")
    )

    # Schema valid (round-trips validate_samples without raising).
    validate_samples(df, name="test")

    # Two RA series (one per category dir), video skipped with warning.
    assert df["series_id"].nunique() == 2
    assert set(df["source"].unique()) == {"real"}

    # Label mapping is correct (counts per file x 2 files).
    counts = df["label"].value_counts().to_dict()
    assert counts.get(FIXATION) == 4  # 2 per file
    assert counts.get(SACCADE) == 4
    assert counts.get(BLINK) == 4
    assert counts.get(OTHER) == 6  # PSO + pursuit + undefined per file = 3
    assert set(df["label"].unique()) <= set(LABELS)

    # t_ms starts at 0 within each series.
    for _, g in df.groupby("series_id"):
        assert g["t_ms"].iloc[0] == pytest.approx(0.0)
        assert g["t_ms"].is_monotonic_increasing

    # Non-NaN pixel inputs yield finite degrees; blink rows are NaN.
    one = df[df["series_id"] == "lund_S1_img_test_labelled_RA"].reset_index(drop=True)
    finite_mask = ~np.isnan(pos[:, 3])
    assert np.isfinite(one.loc[finite_mask, "x_deg"]).all()
    assert np.isfinite(one.loc[finite_mask, "y_deg"]).all()
    assert one.loc[~finite_mask, "x_deg"].isna().all()

    # Degrees match the contract's pix_to_deg geometry exactly.
    exp_x, exp_y = pix_to_deg(
        pos[finite_mask, 3],
        pos[finite_mask, 4],
        screen_res_px=tuple(_SCREEN_RES),
        screen_size_m=tuple(_SCREEN_DIM),
        view_dist_m=_VIEW_DIST,
    )
    np.testing.assert_allclose(one.loc[finite_mask, "x_deg"].to_numpy(), exp_x)
    np.testing.assert_allclose(one.loc[finite_mask, "y_deg"].to_numpy(), exp_y)


def test_real_download_network_gated(tmp_path):
    """Network-gated: download + load real data; skip if GitHub unreachable."""
    try:
        root = download_lund2013(dest_dir=tmp_path / "lund")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"download unavailable: {exc}")

    df = load_lund2013_samples(root=root, annotator="RA")
    validate_samples(df, name="real")
    assert df["series_id"].nunique() > 0
    assert {FIXATION, SACCADE}.issubset(set(df["label"].unique()))
