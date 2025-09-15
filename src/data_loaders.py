# src/data_loaders.py
import os
import yaml
import pyreadr
import pandas as pd
import numpy as np

# Define script base directory for consistent relative paths
def _get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def return_default_config():
    return {
        "paths": {
            "data_dir": "./data",
            "results_dir": "./results",
            "target_tfr_csv": "./data/target_tfrs.csv",
            "midpoints_csv": "./data/midpoints.csv",  # NEW
        },
        "diagnostics": {"print_target_csv": True},
        "projections": {
            "start_year": 2018, "end_year": 2070, "step_years": 5,
            "death_choices": ["EEVV", "censo_2018", "midpoint"],
            "last_observed_year_by_death": {"EEVV": 2023, "censo_2018": 2018, "midpoint": 2018},
            "period_years": 5, "flows_latest_year": 2021,
        },
        "fertility": {
            "default_tfr_target": 1.5, "convergence_years": 50,
            "smoother": {"kind": "exp", "converge_frac": 0.99, "logistic": {"mid_frac": 0.5, "steepness": None}},
        },
        "midpoints": {  # NEW: default EEVV weight if CSV missing / DPTO not found
            "default_eevv_weight": 0.5
        },
        "age_bins": {
            "expected_bins": ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"],
            "order":         ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"],
        },
        "mortality": {"use_ma": True, "ma_window": 5},
        "runs": {
            "mode": "no_draws",
            "no_draws_tasks": [
                {"sample_type": "mid",  "distribution": None, "label": "mid_omissions"},
                {"sample_type": "low",  "distribution": None, "label": "low_omissions"},
                {"sample_type": "high", "distribution": None, "label": "high_omissions"},
            ],
            "draws": {"num_draws": 1000, "dist_types": ["uniform","pert","beta","normal"], "label_pattern": "{dist}_draw_{i}"},
        },
        "unabridging": {"enabled": True},
        "filenames": {"asfr": "asfr.csv", "lt_M": "lt_M_t.csv", "lt_F": "lt_F_t.csv", "lt_T": "lt_T_t.csv"},
    }

def _resolve(ROOT_DIR, p): return os.path.abspath(os.path.join(ROOT_DIR, p))

def _load_config(ROOT_DIR: str, path: str) -> dict:
    """
    Load YAML config if present; otherwise use defaults.
    Returns (cfg, PATHS) where
      - cfg merges user YAML over defaults
      - PATHS resolves relevant file-system paths relative to ROOT_DIR
    """
    if not os.path.exists(path):
        print(f"[config] No config file at {path}; using built-in defaults.")
        return return_default_config(), {
            "data_dir": _resolve(ROOT_DIR, "./data"),
            "results_dir": _resolve(ROOT_DIR, "./results"),
            "target_tfr_csv": _resolve(ROOT_DIR, "./data/target_tfrs.csv"),
            "midpoints_csv": _resolve(ROOT_DIR, "./data/midpoints.csv"),
        }
    with open(path, "r", encoding="utf-8") as fh:
        cfg_user = yaml.safe_load(fh) or {}
    cfg = return_default_config().copy()
    for k, v in cfg_user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            d = cfg[k].copy(); d.update(v); cfg[k] = d
        else:
            cfg[k] = v
    PATHS = {
        "data_dir": _resolve(ROOT_DIR, cfg["paths"]["data_dir"]),
        "results_dir": _resolve(ROOT_DIR, cfg["paths"]["results_dir"]),
        "target_tfr_csv": _resolve(ROOT_DIR, cfg["paths"]["target_tfr_csv"]),
        "midpoints_csv": _resolve(ROOT_DIR, cfg["paths"].get("midpoints_csv", "./data/midpoints.csv"))
    }
    return cfg, PATHS


def allocate_and_drop_missing_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each (DPTO_NOMBRE, SEXO, FUENTE, ANO, VARIABLE) group,
    redistribute the total VALOR of missing-EDAD rows across observed ages
    in proportion to observed VALOR, then drop missing-EDAD rows.
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
        if M > 0 and S > 0:
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

        if asfr_series:
            df_dist = pd.concat(asfr_series, axis=1)
            df_list.append(df_dist)
        tfr_dict[dist] = tfr_values

    stacked_df_all = pd.concat(df_list, axis=1) if df_list else pd.DataFrame()
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

    valid = df[omision_col].notna()
    levels = df.loc[valid, omision_col].astype(int)

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

    return V * (1.0 + eps)
