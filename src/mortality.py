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
    open_interval_width: int = 5
) -> pd.DataFrame:
    """
    Construct an abridged period life table with internal validation.

    Uses the Andreev–Kingkade female formulas for a0 when you supply separate
    0–1 and 1–4 intervals; otherwise uses the constant‐force analytic formula
    on the single first interval [0,n].

    Parameters
    ----------
    ages : array-like or pandas Index/Series
        Left‐endpoints of age intervals (e.g. [0,1,5,10,…] or [0,5,10,…]).
    population, deaths : array-like or pandas Series
        Exposed‐to‐risk and death counts aligned with `ages`.
    radix : int
        Starting cohort size at age 0.
    open_interval_width : int
        Width for the final open interval.

    Returns
    -------
    DataFrame indexed by age, with columns
    [n, mx, ax, qx, px, lx, dx, Lx, Tx, ex].
    """

    # 1) Pre‐processing
    if isinstance(ages, (pd.Index, pd.Series)):
        ages = parse_age_labels(ages)
    df = pd.DataFrame({
        "age": np.asarray(ages, float),
        "population": np.asarray(population, float),
        "deaths": np.asarray(deaths, float)
    })
    df = (
        df.groupby("age", as_index=False)
          .sum()
          .sort_values("age")
          .reset_index(drop=True)
    )
    diffs = np.diff(df["age"].to_numpy())
    if not np.all(diffs > 0):
        raise ValueError("`ages` must be strictly increasing after sorting.")

    # 2) Core computations
    df["n"]  = np.append(diffs, open_interval_width)
    df["mx"] = df["deaths"] / df["population"]
    df["ax"] = 0.5 * df["n"]  # default: uniform within interval

    # ——————————————————————————————
    # Andreev–Kingkade a0 for female if you have 0–1 & 1–4:
    if (
        len(df) >= 3
        and df.loc[0, "age"] == 0.0
        and df.loc[1, "age"] == 1.0
        and df.loc[2, "age"] == 5.0
    ):
        m0 = df.loc[0, "mx"]
        if m0 < 0.01724:
            df.loc[0, "ax"] = 0.14903 - 2.05527 * m0
        elif m0 < 0.06891:
            df.loc[0, "ax"] = 0.04667 + 3.88089 * m0
        else:
            df.loc[0, "ax"] = 0.31411
        # leave a1 = n/2 for ages 1–4 (i.e. 2.0) or override if desired

    # otherwise: constant‐force analytic for single [0,n]
    else:
        n0 = df.loc[0, "n"]
        m0 = df.loc[0, "mx"]
        if m0 > 0:
            df.loc[0, "ax"] = (
                1.0 / m0
                - n0 * np.exp(-m0 * n0)
                  / (1.0 - np.exp(-m0 * n0))
            )

    # qx, px
    df["qx"] = (df["n"] * df["mx"]) / (1.0 + (df["n"] - df["ax"]) * df["mx"])
    df["qx"] = df["qx"].clip(upper=1.0)
    df["px"] = 1.0 - df["qx"]

    # lx
    df["lx"] = np.nan
    df.loc[0, "lx"] = float(radix)
    df.loc[1:, "lx"] = float(radix) * df["px"].iloc[:-1].cumprod().to_numpy()

    # dx, Lx
    df["dx"] = df["lx"] * df["qx"]
    df["Lx"] = df["n"] * df["lx"] - (df["n"] - df["ax"]) * df["dx"]
    last = df.index[-1]
    df.loc[last, "Lx"] = df.loc[last, "lx"] / df.loc[last, "mx"]

    # Tx, ex
    df["Tx"] = df["Lx"][::-1].cumsum()[::-1]
    df["ex"] = df["Tx"] / df["lx"]

    # 3) Post‐validation
    # Check n > 0
    if not (df["n"] > 0).all():
        print("Failing rows for `n` must be positive:")
        print(df.loc[~(df["n"] > 0), ["n"]])
        assert False, "`n` must be positive."

    # Check ax ∈ [0,n]
    if not df["ax"].between(0, df["n"]).all():
        print("Failing rows for `ax` outside [0,n]:")
        print(df.loc[~df["ax"].between(0, df["n"]), ["ax", "n"]])
        assert False, "`ax` outside [0,n]."

    # Check qx ∈ [0,1]
    if not df["qx"].between(0, 1).all():
        print("Failing rows for `qx` outside [0,1]:")
        print(df)
        assert False, "`qx` outside [0,1]."

    # Check px ∈ [0,1]
    if not df["px"].between(0, 1).all():
        print("Failing rows for `px` outside [0,1]:")
        print(df.loc[~df["px"].between(0, 1), ["px"]])
        assert False, "`px` outside [0,1]."

    # Check dx ≥ 0
    if not (df["dx"] >= 0).all():
        print("Failing rows for `dx` negative:")
        print(df.loc[~(df["dx"] >= 0), ["dx"]])
        assert False, "`dx` negative."

    return df.set_index("age")
