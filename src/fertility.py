# src/fertility.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from helpers import _find_col  # moved from here to helpers for reuse


# ------------------------- Target TFR + custom convergence --------------------

def get_target_params(file_path: str) -> Tuple[Dict, Dict]:
    """
    Read target_tfrs.csv and return:
      - targets: dict DPTO_NOMBRE -> Target_TFR
      - conv_years: dict DPTO_NOMBRE -> custom convergence years (int)
    The convergence column is detected liberally (e.g., 'convergeance_year', 'convergence_years', etc.).
    """
    df = pd.read_csv(file_path)
    # DPTO name column (expect 'DPTO_NOMBRE')
    name_col = "DPTO_NOMBRE"
    if name_col not in df.columns:
        cand = _find_col(df, ["dpto", "nombre"])
        if not cand:
            raise KeyError("Could not find 'DPTO_NOMBRE' column in target TFR CSV.")
        name_col = cand

    # Target TFR column
    tfr_col = "Target_TFR"
    if tfr_col not in df.columns:
        cand = _find_col(df, ["tfr"]) or _find_col(df, ["target"])
        if not cand:
            raise KeyError("Could not find 'Target_TFR' column in target TFR CSV.")
        tfr_col = cand

    # Convergence years column (optional)
    conv_col = _find_col(df, ["converge", "year"])
    targets = {}
    conv_years = {}

    for _, r in df.iterrows():
        dpto = str(r[name_col]).strip()
        tfr_val = r[tfr_col]
        try:
            tfr_f = float(tfr_val)
            if np.isfinite(tfr_f):
                targets[dpto] = tfr_f
        except Exception:
            pass
        if conv_col is not None and conv_col in df.columns:
            try:
                cy = int(float(r[conv_col]))
                if cy > 0:
                    conv_years[dpto] = cy
            except Exception:
                pass

    return targets, conv_years


def validate_asfr(asfr: pd.Series, ages, *, warnings_only: bool = True) -> list:
    """
    Validate ASFR for biological plausibility.
    
    Parameters
    ----------
    asfr : pd.Series of age-specific fertility rates (indexed by age labels).
    ages : List-like of age labels.
    warnings_only : If True, return warnings list; if False, raise ValueError on violations.
    
    Returns
    -------
    List of warning strings for implausible values.
    
    Checks
    ------
    1. Reproductive age range (10-55 years)
    2. Maximum biologically possible ASFR (~0.40)
    3. TFR range (0.3 - 10.0)
    4. Peak fertility age (should be 20-35)
    """
    warnings_list = []
    
    # Check 1: Age range (reproductive ages 10-55)
    for age_str in asfr.index:
        try:
            age = int(float(str(age_str).strip()))
            if age < 10 or age > 55:
                if asfr[age_str] > 0.001:  # non-trivial fertility
                    warnings_list.append(
                        f"Age {age}: Fertility rate {asfr[age_str]:.4f} outside "
                        f"reproductive ages (10-55). Biologically implausible."
                    )
        except (ValueError, TypeError):
            pass  # Non-numeric age labels
    
    # Check 2: Maximum rate (biological maximum ~0.40)
    max_asfr = float(asfr.max())
    if max_asfr > 0.40:
        warnings_list.append(
            f"Maximum ASFR {max_asfr:.4f} exceeds biological maximum (~0.40). "
            f"Even high-fertility populations rarely exceed 0.35."
        )
    
    # Check 3: TFR range
    # TFR = sum of ASFR (assumes 1-year age groups; adjust if 5-year)
    tfr = float(asfr.sum())
    if tfr > 10.0:
        warnings_list.append(
            f"TFR {tfr:.2f} exceeds historical maximum (~9-10). "
            f"Check for data errors or improper scaling."
        )
    if tfr > 0 and tfr < 0.30:  # Only warn if non-zero
        warnings_list.append(
            f"TFR {tfr:.2f} below minimum observed in modern populations (~0.8). "
            f"Extreme low fertility - verify data quality."
        )
    
    # Check 4: Peak fertility age (should be 20-35)
    if asfr.max() > 0:
        peak_age_idx = asfr.idxmax()
        try:
            peak_age = int(float(str(peak_age_idx).strip()))
            if peak_age < 18 or peak_age > 40:
                warnings_list.append(
                    f"Peak fertility at age {peak_age} is unusual. "
                    f"Typically peaks at 20-35 years."
                )
        except (ValueError, TypeError):
            pass
    
    if not warnings_only and warnings_list:
        raise ValueError("ASFR validation failed:\n  " + "\n  ".join(warnings_list))
    
    return warnings_list


def compute_asfr(
    ages,
    population,
    births,
    *,
    min_exposure: float = 1e-9,
    nonneg_asfr: bool = True,
    validate: bool = False
):
    """
    Robust ASFR = births / population with label-based alignment.
    
    Parameters
    ----------
    ages : List-like of age labels to consider (e.g., ['15', '16', ..., '49']).
    population : pd.Series or array-like of population counts, indexed by age labels.
    births : pd.Series or array-like of birth counts, indexed by age labels.
    min_exposure : Minimum exposure (population) to consider non-missing (default 1e-9).
    nonneg_asfr : If True, clip ASFR to nonnegative (default True).
    validate : If True, run biological plausibility checks and log warnings (default False).

    Notes
    -----
      - Keeps only ages present in both population and births.
      - Clips births to nonnegative.
      - Exposure <= min_exposure is treated as missing (ASFR -> 0).
      - If validate=True, checks for biologically implausible rates (e.g., age>55, ASFR>0.40).
    """
    # Convert to Series without overwriting indices
    pop = pd.Series(population, copy=False, dtype="float64")
    bth = pd.Series(births,     copy=False, dtype="float64")

    # Normalize labels
    pop.index = pop.index.astype(str).str.strip()
    bth.index = bth.index.astype(str).str.strip()
    desired = pd.Index(ages).astype(str).str.strip()

    # Align by intersection of labels (keeps only bins present in both)
    common = desired.intersection(pop.index).intersection(bth.index)
    # Preserve the desired ordering
    common = desired[desired.isin(common)]

    pop = pop.reindex(common)
    bth = bth.reindex(common)

    # Guardrails
    bth = bth.clip(lower=0.0)                           # no negative births
    pop = pop.where(pop > float(min_exposure), np.nan)  # zero/neg exposures -> NaN

    with np.errstate(divide='ignore', invalid='ignore'):
        asfr = bth / pop
    asfr = asfr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if nonneg_asfr:
        asfr = asfr.clip(lower=0.0)

    result = pd.DataFrame({'population': pop, 'births': bth, 'asfr': asfr}, index=common)
    
    # Validate if requested
    if validate:
        import logging
        warnings_list = validate_asfr(result['asfr'], ages, warnings_only=True)
        if warnings_list:
            for w in warnings_list:
                logging.warning(f"[ASFR Validation] {w}")
    
    return result
