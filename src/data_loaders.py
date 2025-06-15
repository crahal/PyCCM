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

def correct_valor_for_omission(df, valor_col='VALOR', omision_col='OMISION'):
    """
    Apply correction to VALOR based on OMISION level using midpoint multipliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'VALOR' and 'OMISION' columns.
    valor_col : str
        Name of the column containing the values to be corrected.
    omision_col : str
        Name of the column containing the omission level (1–5).

    Returns
    -------
    pd.Series
        Corrected VALOR as a new Series.
    """

    # Define omission midpoints as proportions
    omission_multipliers = {
        1: 0.054,
        2: 0.154,
        3: 0.2525,
        4: 0.351,
        5: 0.450
    }

    return df[valor_col] + (df[valor_col] * df[omision_col].map(omission_multipliers))


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