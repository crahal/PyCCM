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

    Parameters
    ----------
    ages : array-like or pandas Index/Series
        Left-hand end-points of age intervals (0, 5, 10, …).
    population, deaths : array-like or pandas Series
        Exposed population and death counts aligned with `ages`.
    radix : int, default 100_000
        Cohort size at age 0.
    open_interval_width : int, default 5
        Width assumed for the last *open* interval when computing q_x.

    Returns
    -------
    lifetable : pandas.DataFrame
        Indexed by age with columns n, m_x, a_x, q_x, p_x, l_x, d_x, L_x, T_x, e_x.
    """

    # --- 1. Pre-processing --------------------------------------------------
    if isinstance(ages, (pd.Index, pd.Series)):
        ages = parse_age_labels(ages)

    df = pd.DataFrame({
        "age": np.asarray(ages, dtype=float),
        "population": np.asarray(population, dtype=float),
        "deaths": np.asarray(deaths, dtype=float)
    })

    # Sort and aggregate duplicates (if any)
    df = (
        df.groupby("age", as_index=False)
          .sum()
          .sort_values("age")
          .reset_index(drop=True)
    )

    # Validate strict monotonicity
    diffs = np.diff(df["age"].to_numpy())
    if not np.all(diffs > 0):
        raise ValueError("`ages` must be strictly increasing after sorting.")

    # --- 2. Core computations ----------------------------------------------
    df["n"] = np.append(diffs, open_interval_width)

    # Central death rate
    df["mx"] = df["deaths"] / df["population"]

    # Approximate a_x
    df["ax"] = 0.5 * df["n"]
    #df.loc[df["age"] == 0, "ax"] = 0.1  # more accurate for infant mortality

    # q_x with constraint q_x ≤ 1
    df["qx"] = (df["n"] * df["mx"]) / (1 + (df["n"] - df["ax"]) * df["mx"])
    df["qx"] = df["qx"].clip(upper=1.0)

    df["px"] = 1.0 - df["qx"]

    # l_x   (survivors)
    df["lx"] = np.nan  # ensure float dtype
    df.loc[0, "lx"] = float(radix)
    df.loc[df.index[1:], "lx"] = float(radix) * df.loc[df.index[:-1], "px"].cumprod().to_numpy()

    # d_x
    df["dx"] = df["lx"] * df["qx"]

    # L_x : person-years lived in interval
    df["Lx"] = df["n"] * df["lx"] - (df["n"] - df["ax"]) * df["dx"]

    # Last interval: assume constant force of mortality
    df.loc[df.index[-1], "Lx"] = df.loc[df.index[-1], "lx"] / df.loc[df.index[-1], "mx"]

    # T_x  and e_x
    df["Tx"] = df["Lx"][::-1].cumsum()[::-1]
    df["ex"] = df["Tx"] / df["lx"]

    # --- 3. Post-validation -------------------------------------------------
    assert (df["n"] > 0).all(), "`n` must be positive."
    assert (df["ax"].between(0, df["n"])).all(), "`ax` outside [0,n]."
    assert (df["qx"].between(0, 1)).all(), "`qx` outside [0,1]."
    assert (df["px"].between(0, 1)).all(), "`px` outside [0,1]."
    assert (df["dx"] >= 0).all(), "`dx` negative."

    return (
        df.drop(columns=["population", "deaths"])
          .set_index("age")
    )