# src/abridger.py


from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Iterable
import re
import numpy as np
import pandas as pd


SERIES_KEYS_DEFAULT = [
    "DPTO_NOMBRE", "DPTO_CODIGO", "ANO", "SEXO", "VARIABLE", "FUENTE", "OMISION"
]


# ----------------------------- Parsing -----------------------------

_age_pat    = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")   # e.g., 0-4
_open_pat   = re.compile(r"^\s*(\d+)\s*\+\s*$")          # e.g., 80+
_single_pat = re.compile(r"^\s*(\d+)\s*$")               # e.g., 0


def parse_edad(label: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (lo, hi) inclusive; (lo, None) for open-ended; (None, None) if unparsable."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return (None, None)
    t = str(label).strip()
    m = _age_pat.match(t)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if hi < lo:
            lo, hi = hi, lo
        return (lo, hi)
    m = _open_pat.match(t)
    if m:
        return (int(m.group(1)), None)
    m = _single_pat.match(t)
    if m:
        a = int(m.group(1))
        return (a, a)
    return (None, None)


def collapse_convention2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convention 2: label 'a-(a+1)' denotes the one-year bin [a, a+1), i.e., age a only.
    Collapse any (lo, hi) with hi == lo+1 to (lo, lo).
    """
    out = df.copy()
    mask = (
        out["EDAD_MIN"].notna() &
        out["EDAD_MAX"].notna() &
        (out["EDAD_MAX"] == out["EDAD_MIN"] + 1)
    )
    out.loc[mask, "EDAD_MAX"] = out["EDAD_MIN"]
    return out


# ----------------------------- Life table weights for 0..4 -----------------------------


def default_survivorship_0_to_5() -> Dict[int, float]:
    """
    Illustrative survivorship l_x for x=0..5.
    Replace with series-specific values if you have life tables by dept/sex/year.
    """
    l0 = 1.00
    l1 = 0.96                 # ~4% infant mortality
    annual_q = 0.001          # low mortality ages 1-5
    l2 = l1 * (1 - annual_q)
    l3 = l2 * (1 - annual_q)
    l4 = l3 * (1 - annual_q)
    l5 = l4 * (1 - annual_q)
    return {0:l0, 1:l1, 2:l2, 3:l3, 4:l4, 5:l5}


def nLx_1year(lx: Dict[int, float], a0: float = 0.10) -> Dict[int, float]:
    """nLx for ages 0..4 from survivorship l_x and infant a0."""
    L = {0: lx[1] + a0 * (lx[0] - lx[1])}
    for x in range(1, 5):
        L[x] = 0.5 * (lx[x] + lx[x+1])
    return L


def weights_from_nLx(L: Dict[int, float], ages: List[int]) -> np.ndarray:
    w = np.array([max(L.get(a, 0.0), 0.0) for a in ages], dtype=float)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        w = np.ones(len(ages), dtype=float); s = w.sum()
    return w / s


# ----------------------------- Smoothing solver -----------------------------


def _second_diff_matrix(n: int) -> np.ndarray:
    if n < 3:
        return np.zeros((0, n))
    D = np.zeros((n-2, n), dtype=float)
    for i in range(n-2):
        D[i, i]     =  1.0
        D[i, i+1]   = -2.0
        D[i, i+2]   =  1.0
    return D


def _solve_smooth(A: np.ndarray, b: np.ndarray, n: int, ridge: float = 1e-6) -> np.ndarray:
    """
    Minimize ||D x||^2 + ridge ||x||^2 subject to A x = b.
    KKT system solved by least squares; clip negatives to 0 for numerical robustness.
    """
    D = _second_diff_matrix(n)
    Q = D.T @ D + ridge * np.eye(n)
    K = np.block([[2*Q, A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
    rhs = np.concatenate([np.zeros(n), b])
    sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    x = sol[:n]
    return np.clip(x, 0.0, None)


# ----------------------------- Core unabridging -----------------------------


def _apply_infant_adjustment(constraints: List[Tuple[int,int,float]],
                             variable: str,
                             lx: Optional[Dict[int, float]] = None,
                             a0: float = 0.10) -> List[Tuple[int,int,float]]:
    """
    For VARIABLE == 'poblacion_total':
      - If 0–4 and age 0 present, split remainder 1–4 with nLx weights.
      - If 0–4 only, split 0..4 with nLx weights.
      - If 1–4 present, split 1..4 with nLx weights.
    Constraints are (lo, hi, total) with hi==lo for point constraints.
    """
    if variable != "poblacion_total":
        return constraints


    if lx is None:
        lx = default_survivorship_0_to_5()
    L = nLx_1year(lx, a0=a0)


    cons = list(constraints)
    has_pt0 = any((lo == 0 and hi == 0) for (lo, hi, _) in cons)
    has_04  = any((lo == 0 and hi == 4) for (lo, hi, _) in cons)
    has_14  = any((lo == 1 and hi == 4) for (lo, hi, _) in cons)


    if has_04 and has_pt0:
        T04 = sum(v for (lo, hi, v) in cons if (lo == 0 and hi == 4))
        T0  = sum(v for (lo, hi, v) in cons if (lo == 0 and hi == 0))
        T14 = max(T04 - T0, 0.0)
        cons = [(lo, hi, v) for (lo, hi, v) in cons if not (lo == 0 and hi == 4)]
        w = weights_from_nLx(L, [1,2,3,4])
        cons.extend([(a, a, float(wa*T14)) for a, wa in zip([1,2,3,4], w)])


    elif has_04 and not has_pt0:
        T04 = sum(v for (lo, hi, v) in cons if (lo == 0 and hi == 4))
        cons = [(lo, hi, v) for (lo, hi, v) in cons if not (lo == 0 and hi == 4)]
        w = weights_from_nLx(L, [0,1,2,3,4])
        cons.extend([(a, a, float(wa*T04)) for a, wa in zip([0,1,2,3,4], w)])


    if has_14:
        T14 = sum(v for (lo, hi, v) in cons if (lo == 1 and hi == 4))
        cons = [(lo, hi, v) for (lo, hi, v) in cons if not (lo == 1 and hi == 4)]
        w = weights_from_nLx(L, [1,2,3,4])
        cons.extend([(a, a, float(wa*T14)) for a, wa in zip([1,2,3,4], w)])


    return cons


def _unabridge_one_group(g: pd.DataFrame,
                         series_keys: List[str],
                         variable: str,
                         value_col: str,
                         ridge: float = 1e-6) -> pd.DataFrame:
    """Return single-year rows for one (series_keys) group."""
    finite = g[g["EDAD_MAX"].notna()].copy()
    if finite.empty:
        return pd.DataFrame(columns=[*series_keys, "EDAD", value_col])


    # Constraints (lo, hi, total), hi==None (open-ended) skipped downstream
    cons = [(int(r.EDAD_MIN), int(r.EDAD_MAX), float(getattr(r, value_col)))
            for r in finite.itertuples(index=False)]


    # Infant adjustment for poblacion_total
    cons = _apply_infant_adjustment(cons, variable=variable)


    # Age support from constraints
    lo_support = min(lo for (lo, hi, _) in cons if hi is not None)
    hi_support = max(hi for (lo, hi, _) in cons if hi is not None)
    ages = list(range(lo_support, hi_support + 1))


    # Build A,b for constraints
    A_rows, b_vals = [], []
    idx = {a: i for i, a in enumerate(ages)}
    for lo, hi, total in cons:
        if hi is None:
            continue
        row = np.zeros(len(ages), dtype=float)
        if lo == hi:
            if lo in idx:
                row[idx[lo]] = 1.0
        else:
            for a in range(lo, hi + 1):
                if a in idx:
                    row[idx[a]] += 1.0
        if row.sum() > 0:
            A_rows.append(row)
            b_vals.append(total)


    if not A_rows:
        return pd.DataFrame(columns=[*series_keys, "EDAD", value_col])


    A = np.vstack(A_rows)
    b = np.asarray(b_vals, dtype=float)
    x = _solve_smooth(A, b, n=len(ages), ridge=ridge)


    out = pd.DataFrame({
        **{col: g.iloc[0][col] if col in g.columns else np.nan for col in series_keys},
        "EDAD": [str(a) for a in ages],
        value_col: x
    })
    return out


def unabridge_df(df: pd.DataFrame,
                 series_keys: Iterable[str] = SERIES_KEYS_DEFAULT,
                 value_col: str = "VALOR_corrected",
                 ridge: float = 1e-6) -> pd.DataFrame:
    """
    Unabridge a conteos-like DataFrame using 'value_col' for counts.
    Returns a DataFrame with the same series keys and EDAD as single-year strings.
    """
    series_keys = list(series_keys)
    work = df.copy()


    # Parse ages and apply Convention 2 collapsing
    parsed = work["EDAD"].apply(parse_edad)
    work["EDAD_MIN"] = parsed.apply(lambda t: t[0])
    work["EDAD_MAX"] = parsed.apply(lambda t: t[1])
    work = collapse_convention2(work)


    outputs = []
    passthrough = []


    for _, g in work.groupby(series_keys, dropna=False):
        var = g.iloc[0]["VARIABLE"] if "VARIABLE" in g.columns else ""
        # single-year/open-ended passthrough (EDAD_MAX is NaN => open-ended or unparsable)
        open_or_nan = g[g["EDAD_MAX"].isna()].copy()
        if not open_or_nan.empty:
            keep_cols = [*series_keys, "EDAD", value_col]
            for c in keep_cols:
                if c not in open_or_nan.columns:
                    open_or_nan[c] = np.nan
            passthrough.append(open_or_nan[keep_cols].copy())


        out = _unabridge_one_group(g, series_keys, variable=str(var), value_col=value_col, ridge=ridge)
        if not out.empty:
            outputs.append(out)


    single = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=[*series_keys, "EDAD", value_col])
    openp  = pd.concat(passthrough, ignore_index=True) if passthrough else pd.DataFrame(columns=[*series_keys, "EDAD", value_col])
    result = pd.concat([single, openp], ignore_index=True)
    return result


# ----------------------------- Tail harmonization to 90+ -----------------------------


BANDS_70_TO_90 = ["70-74", "75-79", "80-84", "85-89", "90+"]
BANDS_80_TO_90 = ["80-84", "85-89", "90+"]


def _weights_from_pop_or_geometric(
    pop_g: pd.DataFrame,
    bins: List[str],
    pop_value_col: str = "VALOR_corrected",
    r: float = 0.60
) -> np.ndarray:
    """
    Compute weights over 'bins' using population shares if available; otherwise
    use a geometric pattern w_j ∝ r^{j} with j increasing with age (older → smaller).
    """
    v = np.array([float(pop_g.loc[pop_g["EDAD"].astype(str).str.strip().eq(b), pop_value_col].sum())
                  for b in bins], dtype=float)
    S = float(np.nansum(v))
    if np.isfinite(S) and S > 0:
        w = np.nan_to_num(v / S, nan=0.0, posinf=0.0, neginf=0.0)
        s = w.sum()
        if s > 0:
            return w / s
    j = np.arange(len(bins), dtype=float)  # fallback geometric (youngest bin highest)
    w = (r ** j)
    return w / w.sum()


def harmonize_migration_to_90plus(
    mig: pd.DataFrame,
    pop: pd.DataFrame,
    series_keys: List[str],
    value_col: str = "VALOR",                # migration uses VALOR
    pop_value_col: str = "VALOR_corrected"   # population default
) -> pd.DataFrame:
    """
    For each (series_keys) group in 'mig', replace any '70+' with
    70-74,75-79,80-84,85-89,90+ and any '80+' with 80-84,85-89,90+.
    Use population shares at the *same* (series_keys) to allocate totals.
    If population shares are unavailable, use a geometric fallback.


    Returns a new DataFrame with the same schema (series_keys + EDAD + value_col),
    and no raw '70+'/'80+' tails.
    """
    req_mig = set(series_keys + ["EDAD", value_col])
    missing_mig = req_mig - set(mig.columns)
    if missing_mig:
        raise KeyError(f"[harmonize_migration_to_90plus] migration frame missing columns: {missing_mig}")


    if pop_value_col not in pop.columns:
        if "VALOR" in pop.columns:
            pop = pop.copy()
            pop[pop_value_col] = pop["VALOR"]
        else:
            raise KeyError(f"[harmonize_migration_to_90plus] population frame lacks '{pop_value_col}' and 'VALOR'.")


    out_rows: List[Dict] = []


    for keys, g in mig.groupby(series_keys, dropna=False):
        g = g.copy()


        # matching population subset
        pop_g = pop.copy()
        if not isinstance(keys, tuple):
            keys = (keys,)
        for c, v in zip(series_keys, keys):
            pop_g = pop_g[pop_g[c].eq(v)]


        # split 70+
        mask_70p = g["EDAD"].astype(str).str.strip().eq("70+")
        if mask_70p.any():
            total_70p = float(g.loc[mask_70p, value_col].sum())
            g = g.loc[~mask_70p].copy()
            w_70 = _weights_from_pop_or_geometric(pop_g, BANDS_70_TO_90, pop_value_col=pop_value_col)
            for band, w in zip(BANDS_70_TO_90, w_70):
                row = {**{c: v for c, v in zip(series_keys, keys)}}
                row["EDAD"] = band
                row[value_col] = float(total_70p * w)
                out_rows.append(row)


        # split 80+
        mask_80p = g["EDAD"].astype(str).str.strip().eq("80+")
        if mask_80p.any():
            total_80p = float(g.loc[mask_80p, value_col].sum())
            g = g.loc[~mask_80p].copy()
            w_80 = _weights_from_pop_or_geometric(pop_g, BANDS_80_TO_90, pop_value_col=pop_value_col)
            for band, w in zip(BANDS_80_TO_90, w_80):
                row = {**{c: v for c, v in zip(series_keys, keys)}}
                row["EDAD"] = band
                row[value_col] = float(total_80p * w)
                out_rows.append(row)


        # keep remaining rows
        out_rows.extend(g[series_keys + ["EDAD", value_col]].to_dict("records"))


    out = pd.DataFrame(out_rows)
    # collapse duplicates per (series_keys, EDAD)
    out = (
        out.groupby(series_keys + ["EDAD"], dropna=False, as_index=False)[value_col]
           .sum()
    )
    return out


# --- Tail harmonization for conteos (poblacion_total & defunciones) to 90+ ---


_CONTEOS_70_TO_90 = ["70-74", "75-79", "80-84", "85-89", "90+"]
_CONTEOS_80_TO_90 = ["80-84", "85-89", "90+"]


def _geom_weights(bands: list[str], r: float, increasing: bool) -> np.ndarray:
    # bands ordered youngest -> oldest
    j = np.arange(len(bands), dtype=float)
    base = (r ** j) if increasing else (r ** (len(bands)-1-j))
    w = base / base.sum()
    return w.astype(float)


def harmonize_conteos_to_90plus(
    df: pd.DataFrame,
    series_keys: list[str],
    value_col: str = "VALOR_corrected",
    r_pop: float = 0.70,     # population tail: younger>older
    r_deaths: float = 1.45   # deaths tail:    older>younger
) -> pd.DataFrame:
    """
    Replace 70+ / 80+ in poblacion_total and defunciones with
    70-74,75-79,80-84,85-89,90+ (or 80-84,85-89,90+) using geometric weights.
    Totals per (series_keys, VARIABLE) preserved. Other variables pass through.
    """
    need = set(series_keys + ["VARIABLE", "EDAD", value_col])
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"[harmonize_conteos_to_90plus] df missing columns: {missing}")


    # Deduplicate grouping columns to avoid Pandas insertion errors.
    group_top = list(dict.fromkeys(series_keys + ["VARIABLE"]))
    out_rows: List[Dict] = []


    for keys, g in df.groupby(group_top, dropna=False):
        g = g.copy()


        # Recover actual group labels as a dict {col: value}
        if not isinstance(keys, tuple):
            keys = (keys,)
        keyvals = {col: val for col, val in zip(group_top, keys)}
        var_val = str(keyvals["VARIABLE"]).strip()
        var_cmp = var_val.lower()


        if var_cmp not in {"poblacion_total", "defunciones"}:
            out_rows.extend(g.to_dict("records"))
            continue


        # 1) split 70+
        m70 = g["EDAD"].astype(str).str.strip().eq("70+")
        if m70.any():
            tot = float(g.loc[m70, value_col].sum())
            g = g.loc[~m70].copy()
            if var_cmp == "poblacion_total":
                w = _geom_weights(_CONTEOS_70_TO_90, r_pop, increasing=False)
            else:  # defunciones
                w = _geom_weights(_CONTEOS_70_TO_90, r_deaths, increasing=True)
            for band, wi in zip(_CONTEOS_70_TO_90, w):
                row = {**keyvals}
                row["VARIABLE"] = var_val
                row["EDAD"] = band
                row[value_col] = float(tot * wi)
                out_rows.append(row)


        # 2) split 80+
        m80 = g["EDAD"].astype(str).str.strip().eq("80+")
        if m80.any():
            tot = float(g.loc[m80, value_col].sum())
            g = g.loc[~m80].copy()
            if var_cmp == "poblacion_total":
                w = _geom_weights(_CONTEOS_80_TO_90, r_pop, increasing=False)
            else:
                w = _geom_weights(_CONTEOS_80_TO_90, r_deaths, increasing=True)
            for band, wi in zip(_CONTEOS_80_TO_90, w):
                row = {**keyvals}
                row["VARIABLE"] = var_val
                row["EDAD"] = band
                row[value_col] = float(tot * wi)
                out_rows.append(row)


        # 3) keep remaining rows (no 70+/80+)
        out_rows.extend(g.to_dict("records"))


    out = pd.DataFrame(out_rows)
    # Final aggregation with deduplicated groupers.
    group_final = list(dict.fromkeys(series_keys + ["VARIABLE", "EDAD"]))
    out = (
        out.groupby(group_final, dropna=False, as_index=False)[value_col]
           .sum()
    )
    return out


# ----------------------------- Public API -----------------------------


def unabridge_all(*,
                  df: pd.DataFrame,
                  emi: pd.DataFrame,
                  imi: pd.DataFrame,
                  series_keys: Iterable[str] = SERIES_KEYS_DEFAULT,
                  conteos_value_col: str = "VALOR_corrected",
                  ridge: float = 1e-6) -> Dict[str, pd.DataFrame]:
    """
    Unabridge:
      - df (using df[conteos_value_col], typically VALOR_corrected),
      - emi, imi (using their VALOR).
    Returns dict with keys: 'conteos', 'emi', 'imi' (all single-year).
    """
    series_keys = list(series_keys)


    # -------- conteos: use VALOR_corrected if present, else VALOR --------
    if conteos_value_col not in df.columns:
        if "VALOR" in df.columns:
            conteos_value_col = "VALOR"
        else:
            raise KeyError(f"Expected {conteos_value_col!r} (or 'VALOR') in conteos df.")


    conteos_un = unabridge_df(
        df,
        series_keys=series_keys,
        value_col=conteos_value_col,
        ridge=ridge,
    )


    # -------- migration: use VALOR exactly as provided --------
    def _prep_mig(m: pd.DataFrame, varname: str) -> pd.DataFrame:
        m = m.copy()
        for c in series_keys:
            if c not in m.columns:
                m[c] = np.nan
        m["VARIABLE"] = varname
        if "VALOR" not in m.columns:
            raise KeyError(f"Expected 'VALOR' in migration frame for {varname}.")
        return m[[*series_keys, "EDAD", "VALOR"]]


    emi_p = _prep_mig(emi, "flujo_emigracion")
    imi_p = _prep_mig(imi, "flujo_inmigracion")


    emi_un = unabridge_df(emi_p, series_keys=series_keys, value_col="VALOR", ridge=ridge)
    imi_un = unabridge_df(imi_p, series_keys=series_keys, value_col="VALOR", ridge=ridge)


    return {"conteos": conteos_un, "emi": emi_un, "imi": imi_un}


def save_unabridged(objs: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """Persist unabridged outputs to CSV."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for key, frame in objs.items():
        path = os.path.join(out_dir, f"{key}_unabridged_single_year.csv")
        frame.to_csv(path, index=False)


__all__ = [
    "SERIES_KEYS_DEFAULT",
    "parse_edad",
    "collapse_convention2",
    "unabridge_df",
    "unabridge_all",
    "save_unabridged",
    "harmonize_migration_to_90plus",
    "harmonize_conteos_to_90plus",
]
