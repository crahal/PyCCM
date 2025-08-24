import numpy as np
import pandas as pd

def parse_age_labels(age_labels):
    """
    Extract lower bounds of age intervals from strings like '0-4', '90+', etc.
    Returns a list of integers representing starting age of each interval.
    """
    return age_labels.str.extract(r'(\d+)')[0].astype(int)


import numpy as np
import pandas as pd
import warnings

def make_lifetable(
    ages,
    population,
    deaths,
    *,
    radix: int = 100_000,
    open_interval_width: int = 5
) -> pd.DataFrame:
    """
    Abridged period life table. No hard failure on Lx monotonicity:
    the open interval is auto-repaired if needed.
    Returns columns: [n, mx, ax, qx, px, lx, dx, Lx, Tx, ex].
    """
    eps = 1e-12  # numerical floor

    # 1) Inputs & alignment
    if isinstance(ages, (pd.Index, pd.Series)):
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
        warnings.warn("Negative population/deaths encountered; proceeding but results may be invalid.")

    diffs = np.diff(df["age"].to_numpy())
    if not np.all(diffs > 0):
        warnings.warn("Ages not strictly increasing after grouping; attempting to proceed.")
    df["n"] = np.append(diffs, open_interval_width).astype(float)

    # 2) mx = D/E with safe division (0/0 -> 0)
    E = df["E"].to_numpy(float)
    D = df["D"].to_numpy(float)
    df["mx"] = np.divide(D, E, out=np.zeros_like(D, dtype=float), where=E > 0)

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

    # 4) qx, px (clip to [0,1]); force open interval (last) to q=1, p=0
    df["qx"] = (df["n"] * df["mx"]) / (1.0 + (df["n"] - df["ax"]) * df["mx"])
    df["qx"] = df["qx"].clip(0.0, 1.0)
    df["px"] = 1.0 - df["qx"]
    last = df.index[-1]
    df.loc[last, "qx"] = 1.0
    df.loc[last, "px"] = 0.0

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
    # Mass balance usually holds with q_last=1; warn if far off
    if not np.isclose(df["dx"].sum(), float(radix), rtol=0.0, atol=1e-6*radix):
        warnings.warn("Σ d_x deviates from radix; check inputs.")

    return df.set_index("age")
