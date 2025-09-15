# src/fertility.py
import pandas as pd
import numpy as np
from helpers import _find_col  # moved from here to helpers for reuse

# ------------------------- Target TFR + custom convergence --------------------

def get_target_params(file_path: str) -> tuple[dict, dict]:
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


def compute_asfr(
    ages,
    population,
    births,
    *,
    min_exposure: float = 1e-9,
    nonneg_asfr: bool = True
):
    """
    Robust ASFR = births / population with label-based alignment.

    Notes:
      - Keeps only ages present in both population and births.
      - Clips births to nonnegative.
      - Exposure <= min_exposure is treated as missing (ASFR -> 0).
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

    return pd.DataFrame({'population': pop, 'births': bth, 'asfr': asfr}, index=common)
