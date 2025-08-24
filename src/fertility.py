import pandas as pd
import numpy as np

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
    bth = bth.clip(lower=0.0)                         # no negative births
    pop = pop.where(pop > float(min_exposure), np.nan)  # zero/neg exposures -> NaN

    with np.errstate(divide='ignore', invalid='ignore'):
        asfr = bth / pop
    asfr = asfr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if nonneg_asfr:
        asfr = asfr.clip(lower=0.0)

    return pd.DataFrame({'population': pop, 'births': bth, 'asfr': asfr}, index=common)
