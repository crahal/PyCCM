# src/mortality.py
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def parse_age_labels(age_labels):
    """
    Extract lower bounds of age intervals from strings like '0-4', '90+', etc.
    Returns a pandas Series of integers representing starting age of each interval.
    """
    return age_labels.str.extract(r'(\d+)')[0].astype(int)


# ---------------------------------------------------------------------
# Penalized-spline (P-spline–style) graduation on single-year ages
# ---------------------------------------------------------------------
def _expand_closed_intervals(ages: np.ndarray, widths: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand abridged *closed* intervals [a, a+n) into single-year ages a,a+1,...,a+n-1.
    Returns
      single_ages : (N,) float array of single-year ages
      group_id    : (N,) int array mapping each single-year age to its interval index
      closed_idx  : (m-1,) int array of interval indices that are closed (exclude last open)
    """
    ages = np.asarray(ages, float)
    widths = np.asarray(widths, float)
    m = len(ages)
    if m == 0:
        return np.array([]), np.array([]), np.array([])
    closed_idx = np.arange(0, m - 1)  # exclude last (open) interval
    single_ages, group_id = [], []
    for j in closed_idx:
        a = int(round(ages[j]))
        n = int(round(widths[j]))
        single_ages.extend(range(a, a + n))
        group_id.extend([j] * n)
    return np.array(single_ages, dtype=float), np.array(group_id, dtype=int), closed_idx


def _difference_matrix(n: int, order: int) -> np.ndarray:
    """
    Finite-difference operator of given order on an n-vector.
    D (shape (n-order, n)) satisfies (Df)_i = sum_{r=0}^order (-1)^r * C(order, r) * f_{i+order-r}.
    """
    if order < 1:
        raise ValueError("difference order must be >= 1")
    from math import comb
    coeff = np.array([(-1)**r * comb(order, r) for r in range(order + 1)], dtype=float)  # length order+1
    rows = n - order
    D = np.zeros((rows, n), dtype=float)
    for i in range(rows):
        D[i, i:i+order+1] = coeff[::-1]
    return D


def _poisson_pspline_fit(
    E: np.ndarray,
    D: np.ndarray,
    *,
    lam: float = 200.0,
    diff_order: int = 3,
    max_iter: int = 60,
    tol: float = 1e-8,
    verbose: bool = False
) -> np.ndarray:
    """
    Penalized Poisson IRLS on the log-rate f = log m, with roughness penalty λ ||Δ^k f||^2.
    Solves for f ∈ R^n minimizing:
        L(f) = sum_i (E_i * exp(f_i) - D_i * f_i) + (λ/2) ||D_k f||^2,
    where D_k is the k-th order finite-difference operator.
    """
    n = D.size
    if n != E.size:
        raise ValueError("E and D must have the same length.")
    # Initialize f by stabilized log rate
    f = np.log((D + 0.5) / np.maximum(E, 1e-12))

    # Penalty matrices
    Dk = _difference_matrix(n, diff_order)
    P = Dk.T @ Dk  # (n x n), symmetric positive semidefinite

    def obj(fvec):
        mu = E * np.exp(fvec)
        return float(np.sum(mu - D * fvec) + 0.5 * lam * (fvec @ (P @ fvec)))

    last_obj = obj(f)

    for it in range(1, max_iter + 1):
        mu = E * np.exp(f)                   # mean
        g = mu - D                           # gradient (without penalty) wrt f
        H_diag = mu                          # Hessian (diag) from Poisson part
        # Full system: (diag(H_diag) + lam P) Δ = -(g + lam P f)
        rhs = -(g + lam * (P @ f))
        H = np.diag(H_diag) + lam * P

        # Solve for Newton step
        try:
            delta = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            # Add tiny ridge if numerical issue
            H_reg = H + 1e-8 * np.eye(n)
            delta = np.linalg.solve(H_reg, rhs)

        # Backtracking line-search to ensure descent
        step = 1.0
        f_new = f + step * delta
        new_obj = obj(f_new)
        # Simple Armijo-like backtracking
        while not np.isfinite(new_obj) or new_obj > last_obj - 1e-4 * step * float(delta @ (H @ delta)):
            step *= 0.5
            if step < 1e-6:
                break
            f_new = f + step * delta
            new_obj = obj(f_new)

        if verbose:
            print(f"iter={it:02d}  step={step:.3g}  obj={new_obj:.6e}")

        f = f_new
        if np.linalg.norm(delta, ord=np.inf) < tol:
            break
        last_obj = new_obj

    return f


def pspline_group_qx(
    ages: np.ndarray,
    widths: np.ndarray,
    population: np.ndarray,
    deaths: np.ndarray,
    *,
    lam: float = 200.0,
    diff_order: int = 3,
    max_iter: int = 60,
    tol: float = 1e-8
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute *interval* probabilities q_[a,a+n) from a penalized Poisson spline
    fit on single-year ages.

    Steps:
      1) Expand closed intervals uniformly into single-year ages.
      2) Split exposures uniformly: E_{j,k} = E_j / n_j.
         Split observed deaths proportionally: D_{j,k} = D_j / n_j.
         (For Poisson likelihood this preserves the MLE.)
      3) Fit penalized Poisson model for f_k = log m_k on the single-year grid.
      4) Convert to single-year probabilities q_k = 1 - exp(-m_k).
      5) Aggregate to interval probabilities: q_j = 1 - Π_k (1 - q_k) over k ∈ interval j.
         Set last (open) interval q = 1 by convention.
    """
    ages = np.asarray(ages, float)
    widths = np.asarray(widths, float)
    Egrp = np.asarray(population, float)
    Dgrp = np.asarray(deaths, float)
    m = len(ages)

    # 1–2) Expand and split
    single_ages, group_id, closed_idx = _expand_closed_intervals(ages, widths)
    if single_ages.size == 0:
        # Degenerate case: only open interval present
        q_int = np.zeros(m, dtype=float)
        if m >= 1:
            q_int[-1] = 1.0
        return q_int, {"lambda": float(lam), "diff_order": int(diff_order)}

    n_per_group = widths[closed_idx].astype(int)
    E_split = np.repeat(Egrp[closed_idx] / np.maximum(n_per_group, 1), n_per_group)
    D_split = np.repeat(Dgrp[closed_idx] / np.maximum(n_per_group, 1), n_per_group)

    # 3) Fit penalized Poisson spline on f = log m
    f_hat = _poisson_pspline_fit(
        E_split, D_split,
        lam=float(lam), diff_order=int(diff_order),
        max_iter=int(max_iter), tol=float(tol)
    )
    m_single = np.exp(f_hat)                   # hazard per single year
    q_single = 1.0 - np.exp(-m_single)         # single-year death prob (constant force of mort.)

    # 4) Aggregate to interval probabilities
    q_int = np.zeros(m, dtype=float)
    # For each closed interval, take product of survivals
    start = 0
    for j, n in zip(closed_idx, n_per_group):
        block = q_single[start:start+n]
        surv = np.prod(1.0 - block)
        q_int[j] = 1.0 - surv
        start += n
    if m >= 1:
        q_int[-1] = 1.0  # open interval

    return q_int, {"lambda": float(lam), "diff_order": int(diff_order)}


# ---------------------------------------------------------------------
# Abridged period life table (with optional P-spline and Moving Average)
# ---------------------------------------------------------------------
def make_lifetable(
    ages,
    population,
    deaths,
    *,
    radix: int = 100_000,
    open_interval_width: int = 5,
    use_pspline: bool = False,
    pspline_kwargs: Dict[str, float] | None = None,
    use_ma: bool = True,
    ma_window: int = 5
) -> pd.DataFrame:
    """
    Abridged period life table. No hard failure on Lx monotonicity:
    the open interval is auto-repaired if needed.
    Returns columns: [n, mx, ax, qx, px, lx, dx, Lx, Tx, ex].

    Options
    -------
    use_pspline : if True, replace qx with penalized spline graduation.
    use_ma      : if True, replace mx (then qx) with a simple moving-average
                  graduation on log(mx) using a centered window of size `ma_window`.
    """
    eps = 1e-12  # numerical floor

    # 1) Inputs & alignment
    if isinstance(ages, (pd.Index, pd.Series)):
        ages = parse_age_labels(ages)
    df = pd.DataFrame({
        "age": np.asarray(ages, float),
        "E":   np.asarray(population, float),
        "D":   np.asarray(deaths, float),
    })
    df = (df.groupby("age", as_index=False)
            .sum()
            .sort_values("age")
            .reset_index(drop=True))

    if (df[["E","D"]] < 0).any().any():
        warnings.warn("Negative population/deaths encountered; proceeding but results may be invalid.")

    diffs = np.diff(df["age"].to_numpy())
    if not np.all(diffs > 0):
        warnings.warn("Ages not strictly increasing after grouping; attempting to proceed.")
    df["n"] = np.append(diffs, open_interval_width).astype(float)

    # 2) crude mx
    E = df["E"].to_numpy(float)
    D = df["D"].to_numpy(float)
    df["mx"] = np.divide(D, E, out=np.zeros_like(D, dtype=float), where=E > 0)

    # --- Optional: moving average smoothing of mx (on log-scale) ---
    if use_ma and len(df) > 1:
        log_mx = np.log(np.maximum(df["mx"], eps))
        log_mx_smooth = log_mx.rolling(window=int(ma_window), center=True, min_periods=1).mean()
        df["mx"] = np.exp(log_mx_smooth)
        df["ma_window"] = int(ma_window)  # metadata column

    # 3) ax
    df["ax"] = 0.5 * df["n"]
    has_0_1_4 = (
        len(df) >= 3 and df.loc[0, "age"] == 0.0 and
        df.loc[1, "age"] == 1.0 and df.loc[2, "age"] == 5.0
    )
    if has_0_1_4:
        m0 = df.loc[0, "mx"]
        if m0 < 0.01724:
            df.loc[0, "ax"] = 0.14903 - 2.05527 * m0
        elif m0 < 0.06891:
            df.loc[0, "ax"] = 0.04667 + 3.88089 * m0
        else:
            df.loc[0, "ax"] = 0.31411
    else:
        n0 = float(df.loc[0, "n"])
        m0 = float(max(df.loc[0, "mx"], eps))
        df.loc[0, "ax"] = 1.0/m0 - n0/np.expm1(m0 * n0)

    # 4) qx, px (from possibly-smoothed mx), then optionally replace with P-spline qx
    #    Ensure non-terminal qx < 1 to avoid px = 0 cascades in survivorship.
    df["qx"] = (df["n"] * df["mx"]) / (1.0 + (df["n"] - df["ax"]) * df["mx"])
    last = df.index[-1]
    # cap non-terminal qx just below 1
    df.loc[df.index[:-1], "qx"] = np.minimum(df.loc[df.index[:-1], "qx"].to_numpy(float), 1.0 - eps)
    df.loc[last, "qx"] = 1.0
    df["qx"] = df["qx"].clip(0.0, 1.0)
    df["px"] = 1.0 - df["qx"]
    df.loc[last, "px"] = 0.0

    # --- Optional: P-spline smoothing of qx via Poisson penalized fit on single-year ages ---
    if use_pspline:
        if pspline_kwargs is None:
            pspline_kwargs = {}
        qx_ps, meta = pspline_group_qx(
            df["age"].to_numpy(), df["n"].to_numpy(), df["E"].to_numpy(), df["D"].to_numpy(),
            **pspline_kwargs
        )
        qx_ps = np.clip(qx_ps, 0.0, 1.0)
        qx_ps[-1] = 1.0
        # also cap non-terminal P-spline qx a hair below 1
        if qx_ps.size > 1:
            qx_ps[:-1] = np.minimum(qx_ps[:-1], 1.0 - eps)
        df["qx"] = qx_ps
        df["px"] = 1.0 - df["qx"]
        # attach metadata (constant per row, useful for auditing)
        df["pspline_lambda"] = float(meta["lambda"])
        df["pspline_order"]  = int(meta["diff_order"])

    # 5) lx, dx
    df["lx"] = np.nan
    df.loc[0, "lx"] = float(radix)
    if len(df) > 1:
        df.loc[1:, "lx"] = float(radix) * df["px"].iloc[:-1].cumprod().to_numpy()
    df["dx"] = df["lx"] * df["qx"]

    # 6) Lx (closed intervals) and open interval via Lω = lω / mω
    df["Lx"] = df["n"] * df["lx"] - (df["n"] - df["ax"]) * df["dx"]
    df.loc[last, "Lx"] = df.loc[last, "lx"] / max(float(df.loc[last, "mx"]), eps)

    # 7) Repair: enforce L_last ≤ L_prev by increasing m_last if necessary
    if len(df) >= 2:
        prev = last - 1
        if df.loc[last, "Lx"] > df.loc[prev, "Lx"]:
            tiny = 1e-9
            target = max(df.loc[prev, "Lx"] - tiny, tiny)
            new_m_last = df.loc[last, "lx"] / target
            df.loc[last, "mx"] = max(float(df.loc[last, "mx"]), float(new_m_last))
            df.loc[last, "Lx"] = df.loc[last, "lx"] / df.loc[last, "mx"]

    # 8) Tx, ex
    df["Tx"] = df["Lx"][::-1].cumsum()[::-1]
    df["ex"] = df["Tx"] / df["lx"]

    # 9) Soft validations (warnings only)
    if not df["ax"].between(0, df["n"]).all():
        warnings.warn("`ax` outside [0,n] for some ages; results may be unreliable.")
    if not df["qx"].between(0, 1).all():
        warnings.warn("`qx` outside [0,1] after clipping; check inputs.")
    if not df["px"].between(0, 1).all():
        warnings.warn("`px` outside [0,1] after clipping; check inputs.")
    if not np.isclose(df["dx"].sum(), float(radix), rtol=0.0, atol=1e-6*radix):
        warnings.warn("Σ d_x deviates from radix; check inputs.")

    return df.set_index("age")
