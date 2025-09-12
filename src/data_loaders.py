import os
import pyreadr
import pandas as pd
import numpy as np

# Define script base directory for consistent relative paths
def _get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))


def allocate_and_drop_missing_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each (DPTO_NOMBRE, SEXO, FUENTE) group,
    redistribute the total VALOR of missing-EDAD rows across observed ages,
    then drop all rows with missing EDAD.
    """
    df = df.copy()
    grouping = ['DPTO_NOMBRE', 'SEXO', 'FUENTE', 'ANO', 'VARIABLE']

    # 1) redistribute
    for keys, grp in df.groupby(grouping):
        idx = grp.index
        miss = grp['EDAD'].isna()
        obs  = ~miss
        M = grp.loc[miss, 'VALOR'].sum()
        S = grp.loc[obs,  'VALOR'].sum()
#        print(M, keys, grp)
        if M > 0 and S > 0:
            # fractional weights on observed ages
            weights = grp.loc[obs, 'VALOR'] / S
            df.loc[idx[obs], 'VALOR_withmissing'] += M * weights
    # 2) drop all rows with missing EDAD
    df = df[df['EDAD'].notna()].copy()
    return df


def get_lifetables_ex(DPTO):
    """
    Load 'ex' columns from lifetable draw CSVs across specified distributions
    and return a consolidated DataFrame with de-fragmented memory.
    """
    base_dir = _get_base_dir()
    distributions = ["beta", "normal", "pert", "uniform"]
    series_list = []
    for dist in distributions:
        dist_dir = os.path.join(base_dir, "..", "results", "lifetables", DPTO, "draw", dist)
        for file_name in os.listdir(dist_dir):
            file_path = os.path.join(dist_dir, file_name)
            df = pd.read_csv(file_path)
            if 'ex' not in df.columns:
                print(df)
            else:
                series = df["ex"].rename(f"{dist}_{file_name}")
            series_list.append(series)
        ex_df_all = pd.concat(series_list, axis=1)
    return ex_df_all


def get_fertility():
    """
    Load 'asfr' columns from fertility draw CSVs across specified distributions,
    return a consolidated DataFrame of ASFR draws and a DataFrame of TFR values.

    Returns:
        stacked_df_all (pd.DataFrame): Concatenated ASFR series for all distributions.
        tfr_df (pd.DataFrame): Total Fertility Rate (TFR) draws per distribution.
    """
    base_dir = _get_base_dir()
    distributions = ["beta", "normal", "pert", "uniform"]
    df_list = []
    tfr_dict = {}

    for dist in distributions:
        dist_dir = os.path.join(base_dir, "..", "results", "asfr", "total_nacional", "draw", dist)
        if not os.path.isdir(dist_dir):
            tfr_dict[dist] = []
            continue

        asfr_series = []
        tfr_values = []

        for file_name in os.listdir(dist_dir):
            file_path = os.path.join(dist_dir, file_name)
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                df = pd.read_csv(file_path)
                if 'asfr' in df.columns and not df.empty:
                    series = df['asfr'].rename(f"{dist}_{file_name}")
                    asfr_series.append(series)
                    tfr_values.append(5 * series.sum())

        # Concatenate series for this distribution
        if asfr_series:
            df_dist = pd.concat(asfr_series, axis=1)
            df_list.append(df_dist)
        # Store TFR values (even if empty)
        tfr_dict[dist] = tfr_values

    # Concatenate across distributions
    if df_list:
        stacked_df_all = pd.concat(df_list, axis=1)
    else:
        stacked_df_all = pd.DataFrame()

    # Build TFR DataFrame
    tfr_df = pd.DataFrame({dist: pd.Series(vals) for dist, vals in tfr_dict.items()})

    return stacked_df_all, tfr_df


def read_rds_file(file_path: str) -> pd.DataFrame:
    """
    Reads an RDS file and extracts the DataFrame.
    """
    try:
        result = pyreadr.read_r(file_path)
        return result[None]
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_path}: {e}")


def load_all_data(data_dir) -> dict:
    """
    Loads all required RDS files and returns them as a dictionary of DataFrames.
    """
    data_files = {
        'conteos': os.path.join(data_dir, 'conteos.rds'),
#        'migracion_destino_edad': os.path.join(data_dir, 'migracion_destino_edad.rds'),
#        'migracion_origen_destino': os.path.join(data_dir, 'migracion_origen_destino.rds'),
#        'tasas_especificas': os.path.join(data_dir, 'tasas_especificas.rds')
    }

    for name, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    return {name: read_rds_file(path) for name, path in data_files.items()}


def correct_valor_for_omission(
    df: pd.DataFrame,
    sample_type: str,
    distribution: str = None,
    valor_col: str = 'VALOR',
    omision_col: str = 'OMISION'
) -> pd.Series:
    """
    Apply correction to VALOR using OMISION intervals; vectorized implementation
    that gives each row its own independent draw when distribution is specified.
    """
    # 1) Define omission‐interval bounds and mid‐points
    omission_ranges = {
        1: (0.004, 0.104),
        2: (0.105, 0.203),
        3: (0.204, 0.301),
        4: (0.302, 0.400),
        5: (0.401, 0.499),
    }
    midpoints = {i: (low + high) / 2 for i, (low, high) in omission_ranges.items()}

    V   = df[valor_col]
    eps = pd.Series(0.0, index=df.index)

    # mask of rows with an OMISION level
    valid = df[omision_col].notna()
    levels = df.loc[valid, omision_col].astype(int)

    # 2) deterministic case: low / mid / high
    if distribution is None:
        if sample_type == 'low':
            eps_vals = levels.map(lambda i: omission_ranges[i][0])
        elif sample_type == 'mid':
            eps_vals = levels.map(lambda i: midpoints[i])
        elif sample_type == 'high':
            eps_vals = levels.map(lambda i: omission_ranges[i][1])
        else:
            raise ValueError("sample_type must be 'low', 'mid', or 'high'")
        eps.loc[valid] = eps_vals

    # 3) stochastic case: one draw per row
    else:
        dist = distribution.lower()
        # build parameter arrays
        a = levels.map(lambda i: omission_ranges[i][0]).to_numpy()
        b = levels.map(lambda i: omission_ranges[i][1]).to_numpy()
        m = levels.map(lambda i: midpoints[i]).to_numpy()
        k = len(a)

        if dist == 'uniform':
            draws = np.random.uniform(a, b, size=k)

        elif dist == 'pert':
            alpha = 1 + 4 * ((m - a) / (b - a))
            beta  = 1 + 4 * ((b - m) / (b - a))
            draws = np.random.beta(alpha, beta, size=k) * (b - a) + a

        elif dist == 'beta':
            draws = np.random.beta(2, 2, size=k) * (b - a) + a

        elif dist == 'normal':
            sigma = (b - a) / 6
            draws = np.random.normal(loc=m, scale=sigma, size=k)
            # truncate to [a,b]
            mask_bad = (draws < a) | (draws > b)
            while mask_bad.any():
                draws[mask_bad] = np.random.normal(
                    loc=m[mask_bad],
                    scale=sigma[mask_bad]
                )
                mask_bad = (draws < a) | (draws > b)

        else:
            raise ValueError(
                "distribution must be 'uniform', 'pert', 'beta', or 'normal'"
            )

        eps.loc[valid] = draws

    # 4) Return corrected VALOR
    return V * (1.0 + eps)
