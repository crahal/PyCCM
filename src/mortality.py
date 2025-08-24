import numpy as np
import pandas as pd

def parse_age_labels(age_labels):
    """
    Extract lower bounds of age intervals from strings like '0-4', '90+', etc.
    Returns a list of integers representing starting age of each interval.
    """
    return age_labels.str.extract(r'(\d+)')[0].astype(int)


def make_lifetable(
    ages,
    population,
    deaths,
    *,
    radix: int = 100_000,
    open_interval_width: int = 5,
    a0_rule: str = "AK_female",   # {"AK_female", "constant_force"}
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Abridged period life table with strict invariants.

    Returns a DataFrame indexed by age with columns:
    [n, mx, ax, qx, px, lx, dx, Lx, Tx, ex].

    Invariants enforced:
      • n_x > 0 for all closed intervals; last interval is open.
      • q_ω = 1, p_ω = 0, d_ω = l_ω, L_ω = l_ω / max(m_ω, eps).
      • 0 ≤ q_x, p_x ≤ 1; ax ∈ [0, n_x]; L_x nonincreasing in x.
    """

    # --- 1) Inputs & alignment
    if isinstance(ages, (pd.Series, pd.Index)):
        ages = parse_age_labels(ages)  # your helper
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
        raise ValueError("population/deaths must be nonnegative.")
    if ( (df["E"] <= 0) & (df["D"] > 0) ).any():
        # death with zero exposure is undefined m_x
        bad = df.loc[(df["E"] <= 0) & (df["D"] > 0), ["age","E","D"]]
        raise ValueError(f"Positive deaths with zero exposure at ages:\n{bad}")

    # strictly increasing ages
    diffs = np.diff(df["age"].to_numpy())
    if not np.all(diffs > 0):
        raise ValueError("`ages` must be strictly increasing after grouping.")

    # --- 2) Interval widths
    df["n"] = np.append(diffs, open_interval_width).astype(float)
    if not (df["n"].iloc[:-1] > 0).all():
        raise ValueError("All closed-interval widths must be positive.")

    # --- 3) Central death rates with safe division
    # mx := D/E; define 0/0 := 0 (no deaths, no exposure)
    mx = np.divide(df["D"].to_numpy(), df["E"].to_numpy(),
                   out=np.zeros_like(df["D"].to_numpy(), dtype=float),
                   where=df["E"].to_numpy() > 0)
    df["mx"] = mx

    # --- 4) ax initialisation
    df["ax"] = 0.5 * df["n"]  # default

    # A–K a0 if 0–1 and 1–4 are present
    has_0_1_4 = (
        len(df) >= 3 and df.loc[0, "age"] == 0.0 and
        df.loc[1, "age"] == 1.0 and df.loc[2, "age"] == 5.0
    )
    if has_0_1_4 and a0_rule == "AK_female":
        m0 = df.loc[0, "mx"]
        # Andreev–Kingkade (female) piecewise in m0
        if m0 < 0.01724:
            df.loc[0, "ax"] = 0.14903 - 2.05527 * m0
        elif m0 < 0.06891:
            df.loc[0, "ax"] = 0.04667 + 3.88089 * m0
        else:
            df.loc[0, "ax"] = 0.31411
        # leave a1 = n/2 for 1–4 unless you implement its refinement
    else:
        # Constant-force analytic a0 with numerically stable expm1
        n0 = float(df.loc[0, "n"])
        m0 = float(max(df.loc[0, "mx"], eps))
        df.loc[0, "ax"] = 1.0/m0 - n0/np.expm1(m0 * n0)

    # --- 5) qx, px for closed intervals
    df["qx"] = (df["n"] * df["mx"]) / (1.0 + (df["n"] - df["ax"]) * df["mx"])
    df["qx"] = df["qx"].clip(lower=0.0, upper=1.0)
    df["px"] = 1.0 - df["qx"]

    # --- 6) Enforce the open interval conventions (last row)
    last = df.index[-1]
    df.loc[last, "qx"] = 1.0
    df.loc[last, "px"] = 0.0

    # --- 7) l_x, d_x
    df["lx"] = np.nan
    df.loc[0, "lx"] = float(radix)
    if len(df) > 1:
        df.loc[1:, "lx"] = float(radix) * df["px"].iloc[:-1].cumprod().to_numpy()
    df["dx"] = df["lx"] * df["qx"]

    # --- 8) Lx; replace the open interval with l/m
    df["Lx"] = df["n"] * df["lx"] - (df["n"] - df["ax"]) * df["dx"]
    m_last = max(float(df.loc[last, "mx"]), eps)
    df.loc[last, "Lx"] = df.loc[last, "lx"] / m_last

    # --- 9) Tx, ex
    df["Tx"] = df["Lx"][::-1].cumsum()[::-1]
    df["ex"] = df["Tx"] / df["lx"]

    # --- 10) Post-validation
    # ax in [0, n]
    if not df["ax"].between(0, df["n"]).all():
        bad = df.loc[~df["ax"].between(0, df["n"]), ["age","ax","n"]]
        raise AssertionError(f"`ax` outside [0,n] at:\n{bad}")

    # qx, px in [0,1]
    if not df["qx"].between(0, 1).all():
        raise AssertionError("`qx` outside [0,1].")
    if not df["px"].between(0, 1).all():
        raise AssertionError("`px` outside [0,1].")

    # Lx must be nonincreasing in age (period lifetable)
    L = df["Lx"].to_numpy()
    if np.any(np.diff(L) > 1e-10):
        where = np.where(np.diff(L) > 1e-10)[0]
        raise AssertionError(f"`Lx` increases at age indices {where}. "
                             "Check exposures/deaths input.")

    # mass balance with open interval: sum(dx) ≈ radix
    if not np.isclose(df["dx"].sum(), float(radix), rtol=0, atol=1e-6*radix):
        raise AssertionError("Mass balance failed: Σ d_x must equal radix.")

    return df.set_index("age")
