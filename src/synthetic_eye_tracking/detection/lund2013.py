"""Real-data loader for the Lund2013 / Andersson et al. (2017) dataset.

Downloads (and caches) the hand-annotated eye-movement recordings from the
``richardandersson/EyeMovementDetectorEvaluation`` GitHub repository and turns
them into the common *labeled sample frame* defined in
:mod:`synthetic_eye_tracking.detection` so that real and synthetic data share a
single geometry convention and feature/classifier pipeline.

Dataset: Lund2013 / Andersson et al. 2017, GPL-3.0.
"""

from __future__ import annotations

import subprocess
import tarfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from . import LUND_LABEL_MAP, SAMPLE_COLUMNS, pix_to_deg, validate_samples

# GitHub codeload tarball (master branch).
LUND_TARBALL_URL = (
    "https://codeload.github.com/richardandersson/"
    "EyeMovementDetectorEvaluation/tar.gz/refs/heads/master"
)

# Name of the directory the tarball extracts to (GitHub appends the branch).
_EXTRACTED_ROOT_NAME = "EyeMovementDetectorEvaluation-master"

# Relative path inside the repo to the annotated .mat files.
_ANNOTATED_SUBDIR = Path("annotated_data") / "data used in the article"

# Fallbacks when a file is missing geometry metadata.
_DEFAULT_SCREEN_RES = (1024.0, 768.0)
_DEFAULT_SCREEN_DIM = (0.38, 0.30)
_DEFAULT_VIEW_DIST = 0.67


def _download_tarball(dest: Path) -> None:
    """Fetch the tarball to ``dest`` using curl, falling back to urllib."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        subprocess.run(
            ["curl", "-fSL", "-o", str(tmp), LUND_TARBALL_URL],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        # Fall back to urllib if curl is unavailable / fails.
        with urllib.request.urlopen(LUND_TARBALL_URL) as resp:  # noqa: S310
            tmp.write_bytes(resp.read())
    tmp.replace(dest)


def download_lund2013(dest_dir: str | Path = "data/raw/lund2013", *, force: bool = False) -> Path:
    """Download and extract the Lund2013 dataset, returning the extracted root.

    Parameters
    ----------
    dest_dir:
        Directory under which the tarball is cached and extracted.
    force:
        Re-download and re-extract even if a cached copy exists.

    Returns
    -------
    Path
        Path to the extracted repository root
        (``<dest_dir>/EyeMovementDetectorEvaluation-master``).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir / _EXTRACTED_ROOT_NAME

    if root.exists() and not force:
        return root

    tarball = dest_dir / "lund.tar.gz"
    if force or not tarball.exists():
        _download_tarball(tarball)

    # Extract under dest_dir. The tarball's top-level dir is the extracted root.
    with tarfile.open(tarball, "r:gz") as tf:
        # Guard against path traversal in archive members.
        members = []
        for m in tf.getmembers():
            target = (dest_dir / m.name).resolve()
            if not str(target).startswith(str(dest_dir.resolve())):
                raise ValueError(f"unsafe path in tarball: {m.name!r}")
            members.append(m)
        tf.extractall(dest_dir, members=members)  # noqa: S202

    if not root.exists():
        raise FileNotFoundError(
            f"expected extracted root {root} not found after extracting {tarball}"
        )
    return root


def _series_id(mat_path: Path) -> str:
    return f"lund_{mat_path.stem}"


def _as_tuple2(value, default: tuple[float, float]) -> tuple[float, float]:
    try:
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            return float(arr[0]), float(arr[1])
    except (TypeError, ValueError):
        pass
    return default


def _as_float(value, default: float) -> float:
    try:
        v = float(np.asarray(value, dtype=float).ravel()[0])
        if np.isfinite(v):
            return v
    except (TypeError, ValueError, IndexError):
        pass
    return default


def _load_one(mat_path: Path) -> pd.DataFrame | None:
    """Parse a single annotated .mat file into a labeled sample frame."""
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    et = mat["ETdata"]
    pos = np.asarray(et.pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] < 6:
        raise ValueError(f"unexpected pos shape {pos.shape}")

    screen_res = _as_tuple2(getattr(et, "screenRes", None), _DEFAULT_SCREEN_RES)
    screen_dim = _as_tuple2(getattr(et, "screenDim", None), _DEFAULT_SCREEN_DIM)
    view_dist = _as_float(getattr(et, "viewDist", None), _DEFAULT_VIEW_DIST)

    ts_us = pos[:, 0]
    x_pix = pos[:, 3]
    y_pix = pos[:, 4]
    labels_int = pos[:, 5]

    t_ms = (ts_us - ts_us[0]) / 1000.0
    x_deg, y_deg = pix_to_deg(
        x_pix,
        y_pix,
        screen_res_px=screen_res,
        screen_size_m=screen_dim,
        view_dist_m=view_dist,
    )

    labels = pd.Series(labels_int).round().astype("Int64").map(LUND_LABEL_MAP)
    if labels.isna().any():
        unknown = sorted(set(np.unique(labels_int[labels.isna().to_numpy()])))
        raise ValueError(f"unmapped integer labels: {unknown}")

    return pd.DataFrame(
        {
            "series_id": _series_id(mat_path),
            "source": "real",
            "t_ms": t_ms,
            "x_deg": x_deg,
            "y_deg": y_deg,
            "label": labels.astype(str).to_numpy(),
        }
    )


def load_lund2013_samples(
    root: str | Path | None = None,
    *,
    annotator: str = "RA",
    categories: tuple[str, ...] = ("img", "dots", "video"),
) -> pd.DataFrame:
    """Load the Lund2013 annotated recordings as a labeled sample frame.

    Parameters
    ----------
    root:
        Extracted dataset root. If ``None``, :func:`download_lund2013` is called.
    annotator:
        Annotator suffix to select (e.g. ``"RA"`` or ``"MN"``).
    categories:
        Stimulus categories (subdirectories) to include.

    Returns
    -------
    pandas.DataFrame
        Frame validated against ``SAMPLE_COLUMNS`` with labels in ``LABELS``.
    """
    if root is None:
        root = download_lund2013()
    root = Path(root)

    base = root / _ANNOTATED_SUBDIR
    if not base.exists():
        # Be tolerant of a root that already points at the annotated tree, or a
        # flat test layout: search anywhere under root.
        base = root

    frames: list[pd.DataFrame] = []
    for category in categories:
        cat_dir = base / category
        if not cat_dir.exists():
            warnings.warn(f"category dir not found: {cat_dir}", stacklevel=2)
            continue
        mat_files = sorted(cat_dir.glob(f"*_labelled_{annotator}.mat"))
        for mat_path in mat_files:
            try:
                df = _load_one(mat_path)
            except Exception as exc:  # noqa: BLE001 - tolerate per-file failures
                warnings.warn(f"skipping {mat_path.name}: {exc}", stacklevel=2)
                continue
            if df is not None and len(df):
                frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"no annotated .mat files found under {base} for annotator={annotator!r} "
            f"categories={categories!r}"
        )

    combined = pd.concat(frames, ignore_index=True)
    return validate_samples(combined, name="lund2013")
