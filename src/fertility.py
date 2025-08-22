import pandas as pd

def compute_asfr(ages, population, births):
    r"""
    Compute Age-Specific Fertility Rates (ASFR) for a given set of age groups.

    Parameters
    ----------
    ages : array-like
        Sequence of age group labels or lower bounds (e.g., 15, 20, ..., 45) used as index.
    population : array-like
        Female population counts (exposure) for each corresponding age group.
    births : array-like
        Number of live births attributed to each corresponding age group.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by age group with the following columns:
        - 'population': female population counts
        - 'births': number of births
        - 'asfr': age-specific fertility rate, computed as:

        \[
            \mathrm{ASFR}_x = \frac{B_x}{F_x}
        \]

        where \(B_x\) is the number of births to women aged \(x\)–\(x+n\),
        and \(F_x\) is the number of women in the same age group.

    Notes
    -----
    The resulting ASFRs represent the number of births per woman in each age group
    during the observation period. This formulation assumes all input vectors are aligned
    and of equal length.
    """
    df = pd.DataFrame(
        {'population': population, 'births': births},
        index=ages
    )
    df['asfr'] = df['births'] / df['population']
    df['asfr'] = df['asfr'].fillna(0)

#    tfr = 5 * df['asfr'].sum()
#    print(f"Estimated TFR: {tfr:.2f}")
#    print("Total corrected births:", df['births'].sum())
#    print("Total female population 15–49:", df['population'].sum())
    return df
