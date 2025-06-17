import pyreadr
import pandas as pd
import numpy as np
import os

def read_rds_file(file_path: str):
    """
    Reads an RDS file and extracts the DataFrame.
    Handles errors gracefully and provides debug information.
    """
    try:
        result = pyreadr.read_r(file_path)
        return result[None]  # Modify as required if other keys exist
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_path}: {e}")

def load_all_data():
    """
    Loads all required RDS files and returns them as a dictionary of DataFrames.
    Handles file validation and path resolution.
    """
    base_dir = os.getcwd()  # Directory of this script
    data_files = {
        'conteos': os.path.join(base_dir, '../data/conteos.rds'),
        'migracion_destino_edad': os.path.join(base_dir, '../data/migracion_destino_edad.rds'),
        'migracion_origen_destino': os.path.join(base_dir, '../data/migracion_origen_destino.rds'),
        'tasas_especificas': os.path.join(base_dir, '../data/tasas_especificas.rds')
    }

    # Validate file existence
    for name, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Load data
    return {
        name: read_rds_file(file_path)
        for name, file_path in data_files.items()
    }

def correct_valor_for_omission(
    df: pd.DataFrame,
    sample_type: str,
    distribution: str = None,
    valor_col: str = 'VALOR',
    omision_col: str = 'OMISION'
) -> pd.Series:
    """
    Apply a correction to VALOR based on OMISION uncertainty intervals.

    Missing OMISION values are assumed to be zero (no correction).

    For omission level i ∈ {1,…,5}, let [a_i, b_i] be the uncertainty interval
    (in proportions). Denote midpoint m_i = (a_i + b_i)/2.

    If distribution is None:
      - sample_type='low'  → ε_i = a_i
      - sample_type='mid'  → ε_i = m_i
      - sample_type='high' → ε_i = b_i

    If distribution is specified, we draw ε_i from one of these within [a_i, b_i]:
      - 'uniform':       Uniform(a_i, b_i)
      - 'pert':          Beta-PERT with mode=m_i
      - 'beta':          Symmetric Beta(α=2, β=2)
      - 'normal':        Truncated Normal(μ=m_i, σ=(b_i-a_i)/6)

    Then V_corr = V_orig * (1 + ε_i). For missing OMISION, ε_i=0.
    """
    # Define intervals [a_i, b_i]
    omission_ranges = {
        1: (0.004, 0.104),
        2: (0.105, 0.203),
        3: (0.204, 0.301),
        4: (0.302, 0.400),
        5: (0.401, 0.499),
    }
    # Compute midpoints
    midpoints = {i: (low + high) / 2 for i, (low, high) in omission_ranges.items()}

    # Extract original values
    V = df[valor_col]
    n = len(df)
    eps = pd.Series(0.0, index=df.index)

    # Identify rows with valid omission levels
    valid = df[omision_col].notna()
    levels = df.loc[valid, omision_col].astype(int)

    # Determine epsilon for valid rows
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
    else:
        dist = distribution.lower()
        a = levels.map(lambda i: omission_ranges[i][0]).to_numpy()
        b = levels.map(lambda i: omission_ranges[i][1]).to_numpy()
        m = levels.map(lambda i: midpoints[i]).to_numpy()
        k = len(levels)
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
            mask = (draws < a) | (draws > b)
            while mask.any():
                new = np.random.normal(loc=m[mask], scale=sigma[mask])
                draws[mask] = new
                mask = (draws < a) | (draws > b)
        else:
            raise ValueError("distribution must be 'uniform', 'pert', 'beta', or 'normal'")
        eps.loc[valid] = draws

    # Compute corrected values with zero correction for missing
    V_corr = V * (1 + eps)
    return V_corr


def read_rds_file(file_path: str):
    """
    Reads an RDS file and extracts the DataFrame.
    Handles errors gracefully and provides debug information.
    """
    try:
        result = pyreadr.read_r(file_path)
        return result[None]  # Modify as required if other keys exist
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_path}: {e}")

def load_all_data():
    """
    Loads all required RDS files and returns them as a dictionary of DataFrames.
    Handles file validation and path resolution.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
    data_files = {
        'conteos': os.path.join(base_dir, '../data/conteos.rds'),
        'migracion_destino_edad': os.path.join(base_dir, '../data/migracion_destino_edad.rds'),
        'migracion_origen_destino': os.path.join(base_dir, '../data/migracion_origen_destino.rds'),
        'tasas_especificas': os.path.join(base_dir, '../data/tasas_especificas.rds')
    }

    # Validate file existence
    for name, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Load data
    return {
        name: read_rds_file(file_path)
        for name, file_path in data_files.items()
    }