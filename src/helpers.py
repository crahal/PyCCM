# src/helpers.py
"""
General-purpose helpers shared across the pipeline.

This module centralizes reusable utilities that are agnostic to domain specifics:
- Age-bin utilities (bin width, vectorized widths, single-year bins).
- Liberal CSV header detection and midpoint weights reader.
- List/string coercions for config values.
- Filename suffix manipulation.
- Fertility/TFR helpers (TFR integral, weight normalization, smoothers).
- Series/index alignment to a provided EDAD ordering.
- Abridged-only pre-omission collapsing of defunciones age groups.

All functions are pure and side-effect free (except for file I/O in CSV readers),
facilitating reuse and unit testing.

IMPORTANT: This module does not import project-specific modules to avoid circular
dependencies. Callers must supply any configuration defaults they need.
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Age scaffolding
# ---------------------------------------------------------------------------

def _single_year_bins() -> list[str]:
    """
    Return single-year age labels '0', '1', ..., '89', '90+'.

    Returns
    -------
    list[str]
        Age labels for single-year ages up to an open-ended 90+ tail.
    """
    return [str(a) for a in range(0, 90)] + ["90+"]


def _bin_width(label: str) -> float:
    """
    Closed-interval bin width from an age label.

    Conventions
    -----------
    - 'lo-hi' → (hi - lo + 1)
    - 'X+'    → 5  (treat the open interval as width 5 for weighting purposes)
    - 'k'     → 1

    Parameters
    ----------
    label : str
        Age label.

    Returns
    -------
    float
        Effective width.
    """
    s = str(label)
    if "-" in s:
        lo, hi = s.split("-")
        return float(hi) - float(lo) + 1.0
    if s.endswith("+"):
        return 5.0
    return 1.0


def _widths_from_index(idx) -> np.ndarray:
    """
    Vectorized bin widths for an index of age labels.

    Parameters
    ----------
    idx : Iterable
        Iterable of age labels.

    Returns
    -------
    np.ndarray
        Widths aligned to `idx`.
    """
    return np.array([_bin_width(x) for x in idx], dtype=float)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, must_include: list[str]) -> str | None:
    """
    Return the first column name in `df` whose lowercase name contains *all*
    substrings in `must_include`. Used for robust header detection.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with candidate columns.
    must_include : list[str]
        Substrings that must all appear in the lowercase column name.

    Returns
    -------
    str | None
        Original column name or None if not found.
    """
    low = {c.lower(): c for c in df.columns}
    for lc, orig in low.items():
        if all(s in lc for s in must_include):
            return orig
    return None


def get_midpoint_weights(file_path: str) -> dict:
    """
    Read a CSV of per-DPTO midpoint weights and return mapping:
        DPTO_NOMBRE -> weight in [0, 1].

    Column detection is liberal:
    - Department name column: 'DPTO_NOMBRE' or any column containing both
      'dpto' and 'nombre' (or 'depart').
    - Value column: first matching one of
        * contains 'mid'
        * contains both 'eevv' and 'weight'
        * contains 'weight'
      otherwise the first numeric column (excluding the name column).

    Parameters
    ----------
    file_path : str
        Path to the CSV.

    Returns
    -------
    dict
        Mapping {department_name: weight ∈ [0,1]}.
    """
    if not os.path.exists(file_path):
        return {}
    df = pd.read_csv(file_path)

    # Department column
    name_col = "DPTO_NOMBRE"
    if name_col not in df.columns:
        cand = _find_col(df, ["dpto", "nombre"]) or _find_col(df, ["depart"])
        if not cand:
            raise KeyError("Could not find 'DPTO_NOMBRE' column in midpoints CSV.")
        name_col = cand

    # Value column
    val_col = (
        _find_col(df, ["mid"])
        or _find_col(df, ["eevv", "weight"])
        or _find_col(df, ["weight"])
    )
    if val_col is None:
        numeric_cols = [
            c for c in df.columns
            if c != name_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if numeric_cols:
            val_col = numeric_cols[0]
        else:
            raise KeyError("Could not infer midpoint value column in midpoints CSV.")

    out: dict[str, float] = {}
    for _, r in df.iterrows():
        dpto = str(r[name_col]).strip()
        try:
            v = float(r[val_col])
        except Exception:
            continue
        if np.isfinite(v):
            out[dpto] = float(np.clip(v, 0.0, 1.0))
    return out


# ---------------------------------------------------------------------------
# List / string coercions for config-like values
# ---------------------------------------------------------------------------

def _coerce_list(x):
    """
    Coerce input to a flat list of strings.

    Rules
    -----
    - If `x` is a list, flatten one level; split any string items on ';' or ','.
    - If `x` is a string, split on ';' or ',' and strip.
    - Otherwise return None (caller should fall back to project defaults).

    Parameters
    ----------
    x : Any

    Returns
    -------
    list[str] | None
    """
    if isinstance(x, list):
        flat: list[str] = []
        for it in x:
            if isinstance(it, list):
                flat.extend(it)
            elif isinstance(it, str) and (";" in it or "," in it):
                flat.extend(
                    [s.strip() for s in it.replace(",", ";").split(";") if s.strip()]
                )
            else:
                flat.append(str(it))
        return flat
    if isinstance(x, str):
        if ";" in x or "," in x:
            return [s.strip() for s in x.replace(",", ";").split(";") if s.strip()]
        return [x.strip()]
    return None


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _with_suffix(fname: str, suffix: str) -> str:
    """
    Insert a suffix before the file extension.

    Example
    -------
    _with_suffix("foo.csv", "_bar") -> "foo_bar.csv"
    """
    if not suffix:
        return fname
    base, ext = os.path.splitext(fname)
    return f"{base}{suffix}{ext}"


# ---------------------------------------------------------------------------
# Fertility / TFR helpers
# ---------------------------------------------------------------------------

def _tfr_from_asfr_df(asfr_df: pd.DataFrame) -> float:
    """
    Compute TFR as the width-weighted sum of ASFR over the index ages.

    Parameters
    ----------
    asfr_df : pd.DataFrame
        Must contain column 'asfr'; index are age labels.

    Returns
    -------
    float
        Total Fertility Rate.
    """
    s = asfr_df["asfr"].astype(float)
    widths = _widths_from_index(s.index)
    return float(np.sum(s.values * widths))


def _normalize_weights_to(idx, w: pd.Series) -> pd.Series:
    """
    Normalize ASFR weights so that sum_i w_i * width_i = 1 over `idx`.

    Parameters
    ----------
    idx : Iterable
        Target age labels.
    w : pd.Series
        Weights indexed by age labels (need not match exactly).

    Returns
    -------
    pd.Series
        Weights reindexed to `idx` and normalized by width.
    """
    idx = pd.Index(idx).astype(str).str.strip()
    w = pd.Series(w, copy=False)
    w.index = w.index.astype(str).str.strip()
    w = w.reindex(idx).fillna(0.0)
    widths = _widths_from_index(idx)
    s = float(np.sum(w.values * widths))
    if not np.isfinite(s) or s <= 0:
        w = pd.Series(1.0, index=idx) / len(idx)
        s = float(np.sum(w.values * widths))
    return w / s


def _exp_tfr(
    TFR0: float, target: float, years: int, step: int, converge_frac: float = 0.99
) -> float:
    """
    Exponential approach to target: gap_t = gap_0 * exp(-kappa t),
    where kappa is chosen so that after `years` we have reached
    `converge_frac` of the way to the target.

    Returns
    -------
    float
        TFR at the given step.
    """
    t = max(step, 0)
    gap0 = float(TFR0 - target)
    if gap0 == 0.0:
        return float(target)
    eps = 1e-12
    q = max(1.0 - converge_frac, eps)
    kappa = -np.log(q) / max(years, 1)
    return float(target + gap0 * np.exp(-kappa * t))


def _logistic_tfr(
    TFR0: float,
    target: float,
    years: int,
    step: int,
    mid_frac: float = 0.5,
    steepness: float | None = None,
) -> float:
    """
    Logistic approach to target with midpoint at `mid_frac * years`.

    If `steepness` is None, it is set to a calibrated value that gives a
    reasonably steep but smooth approach over the horizon.

    Returns
    -------
    float
        TFR at the given step.
    """
    t = max(step, 0.0)
    T = max(years, 1)
    t0 = float(mid_frac) * T
    if steepness is None:
        # Choose s so that the curve transitions over the horizon.
        r = 100.0
        steepness = (np.log(r) - np.log(1 + np.exp(-t0))) / max(T - t0, 1e-9)
    s = float(steepness)
    gap0 = float(TFR0 - target)
    scale = 1.0 + np.exp(-s * t0)
    return float(target + gap0 * (scale / (1.0 + np.exp(s * (t - t0)))))


def _smooth_tfr(
    TFR0: float, target: float, years: int, step: int, kind: str = "exp", **kwargs
) -> float:
    """
    Dispatcher for TFR smoothing paths.

    Parameters
    ----------
    TFR0 : float
        Baseline TFR.
    target : float
        Target TFR.
    years : int
        Convergence horizon in years.
    step : int
        Years elapsed since baseline.
    kind : {"exp", "logistic"}
        Path family.
    kwargs : dict
        Additional parameters for the chosen family.

    Returns
    -------
    float
        Smoothed TFR at `step`.
    """
    if kind == "exp":
        return _exp_tfr(TFR0, target, years, step, **kwargs)
    if kind == "logistic":
        return _logistic_tfr(TFR0, target, years, step, **kwargs)
    raise ValueError(f"Unknown TFR path kind: {kind!r}")


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def fill_missing_age_bins(s: pd.Series, edad_order: list[str]) -> pd.Series:
    """
    Reindex a Series of counts to the provided EDAD order, filling missing with 0.

    Parameters
    ----------
    s : pd.Series
        Series indexed by age labels.
    edad_order : list[str]
        Target EDAD ordering.

    Returns
    -------
    pd.Series
        Reindexed series with float dtype.
    """
    return s.reindex(edad_order, fill_value=0.0).astype(float)


def _ridx(s_like, edad_order: list[str]) -> pd.Series:
    """
    Normalize labels and reindex a Series-like object to EDAD order.

    Steps
    -----
    - Cast index to stripped strings.
    - Aggregate duplicate labels by summation.
    - Reindex to `edad_order` with zeros for missing.

    Parameters
    ----------
    s_like : Mapping or pd.Series
        Age-labeled values.
    edad_order : list[str]
        Target EDAD ordering.

    Returns
    -------
    pd.Series
        Values aligned to `edad_order`.
    """
    s = pd.Series(s_like, copy=False)
    s.index = pd.Index(map(str, s.index)).str.strip()
    if s.index.has_duplicates:
        s = s.groupby(level=0, sort=False).sum()
        s.index = pd.Index(map(str, s.index)).str.strip()
    return s.reindex(edad_order, fill_value=0.0).astype(float)


# ---------------------------------------------------------------------------
# Abridged-only: collapse 0–1 + 2–4 => 0–4 for defunciones BEFORE omission
# ---------------------------------------------------------------------------

_AGE_01_PAT = re.compile(r"^\s*0\s*[-–]\s*1\s*$")
_AGE_24_PAT = re.compile(r"^\s*2\s*[-–]\s*4\s*$")

def _collapse_defunciones_01_24_to_04(df_def: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse '0-1' and '2-4' into a single '0-4' bin for defunciones BEFORE
    applying omission corrections.

    The collapse is performed within groups defined by:
      (DPTO_NOMBRE, DPTO_CODIGO, ANO, SEXO, VARIABLE, FUENTE)

    Rules
    -----
    - VALOR and VALOR_withmissing are summed across the two source bins when present.
    - OMISION is set to the maximum level observed among the two source bins.
    - VALOR_corrected is left as NaN (it is computed later in the pipeline).

    Parameters
    ----------
    df_def : pd.DataFrame
        Subset of conteos for VARIABLE == 'defunciones'.

    Returns
    -------
    pd.DataFrame
        Frame with '0-4' in place of any '0-1'/'2-4' pair, preserving other rows.
    """
    if df_def.empty:
        return df_def.copy()

    df = df_def.copy()
    is_01 = df["EDAD"].astype(str).str.match(_AGE_01_PAT)
    is_24 = df["EDAD"].astype(str).str.match(_AGE_24_PAT)
    needs_collapse = is_01 | is_24
    if not needs_collapse.any():
        return df

    work = df.loc[needs_collapse].copy()

    gkeys = ["DPTO_NOMBRE", "DPTO_CODIGO", "ANO", "SEXO", "VARIABLE", "FUENTE"]
    agg_map: dict[str, str] = {}
    if "VALOR" in work.columns:
        agg_map["VALOR"] = "sum"
    if "VALOR_withmissing" in work.columns:
        agg_map["VALOR_withmissing"] = "sum"

    work["_OM_NUM"] = pd.to_numeric(work.get("OMISION", np.nan), errors="coerce")

    summed = (
        work.groupby(gkeys, dropna=False)
            .agg({**agg_map, "_OM_NUM": "max"})
            .reset_index()
    )

    # Build collapsed rows with the same columns as df
    collapsed = pd.DataFrame(columns=df.columns)
    for k in gkeys:
        collapsed[k] = summed[k]
    collapsed["EDAD"] = "0-4"

    if "VALOR" in df.columns and "VALOR" in summed.columns:
        collapsed["VALOR"] = summed["VALOR"]
    if "VALOR_withmissing" in df.columns and "VALOR_withmissing" in summed.columns:
        collapsed["VALOR_withmissing"] = summed["VALOR_withmissing"]
    if "VALOR_corrected" in df.columns:
        collapsed["VALOR_corrected"] = np.nan
    if "OMISION" in df.columns:
        collapsed["OMISION"] = summed["_OM_NUM"]

    out = pd.concat([df.loc[~needs_collapse], collapsed], ignore_index=True, sort=False)
    return out
