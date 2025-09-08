# src/run_projections.py
# Script rewritten to load hard-coded parameters from a YAML config at project root (../config.yaml).

import os
import sys
import zlib
import yaml
import pyreadr  # noqa: F401 (import kept if used elsewhere)
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from mortality import make_lifetable
from fertility import compute_asfr
from migration import create_migration_frame  # noqa: F401 (import kept if used elsewhere)
from projections import make_projections, save_LL, save_projections
from data_loaders import load_all_data, correct_valor_for_omission, allocate_and_drop_missing_age


# ------------------------------- Config loading -------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")

_DEFAULT_CFG = {
    "paths": {
        "data_dir": "../data",
        "results_dir": "../results",
        "target_tfr_csv": "../data/target_tfrs.csv",
    },
    "diagnostics": {"print_target_csv": True},
    "projections": {
        "start_year": 2018,
        "end_year": 2070,
        "step_years": 5,
        "death_choices": ["EEVV", "censo_2018", "midpoint"],
        "last_observed_year_by_death": {"EEVV": 2023, "censo_2018": 2018, "midpoint": 2018},
        "period_years": 5,
        "flows_latest_year": 2021,
    },
    "fertility": {
        "default_tfr_target": 1.5,
        "convergence_years": 50,
        "smoother": {
            "kind": "exp",          # "exp" | "logistic"
            "converge_frac": 0.99,  # for exp
            "logistic": {"mid_frac": 0.5, "steepness": None},
        },
    },
    "age_bins": {
        "expected_bins": ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
                          "40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"],
        "order": ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
                  "40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"],
    },
    "runs": {
        "mode": "no_draws",  # "no_draws" | "draws"
        "no_draws_tasks": [
            {"sample_type": "mid",  "distribution": None, "label": "mid_omissions"},
            {"sample_type": "low",  "distribution": None, "label": "low_omissions"},
            {"sample_type": "high", "distribution": None, "label": "high_omissions"},
        ],
        "draws": {
            "num_draws": 1000,
            "dist_types": ["uniform", "pert", "beta", "normal"],
            "label_pattern": "{dist}_draw_{i}",
        },
    },
    "parallel": {"processes": 1},
    "filenames": {"asfr": "asfr.csv", "lt_M": "lt_M_t.csv", "lt_F": "lt_F_t.csv", "lt_T": "lt_T_t.csv"},
}

def _load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[config] No config file at {path}; using built-in defaults.")
        return _DEFAULT_CFG
    with open(path, "r", encoding="utf-8") as fh:
        cfg_user = yaml.safe_load(fh) or {}
    # shallow merge: user overrides defaults
    cfg = _DEFAULT_CFG.copy()
    for k, v in cfg_user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            d = cfg[k].copy()
            d.update(v)
            cfg[k] = d
        else:
            cfg[k] = v
    return cfg

CFG = _load_config(CONFIG_PATH)

# Normalise/resolve paths w.r.t. ROOT_DIR
def _resolve(p): return os.path.abspath(os.path.join(ROOT_DIR, p))
PATHS = {
    "data_dir": _resolve(CFG["paths"]["data_dir"]),
    "results_dir": _resolve(CFG["paths"]["results_dir"]),
    "target_tfr_csv": _resolve(CFG["paths"]["target_tfr_csv"]),
}

PRINT_TARGET_CSV = bool(CFG.get("diagnostics", {}).get("print_target_csv", True))

PROJ = CFG["projections"]
START_YEAR = int(PROJ["start_year"])
END_YEAR = int(PROJ["end_year"])
STEP_YEARS = int(PROJ["step_years"])
DEATH_CHOICES = list(PROJ["death_choices"])
LAST_OBS_YEAR = dict(PROJ["last_observed_year_by_death"])
PERIOD_YEARS = int(PROJ["period_years"])
FLOWS_LATEST_YEAR = int(PROJ["flows_latest_year"])

FERT = CFG["fertility"]
DEFAULT_TFR_TARGET = float(FERT["default_tfr_target"])
CONV_YEARS = int(FERT["convergence_years"])
SMOOTHER = FERT["smoother"]
SMOOTH_KIND = SMOOTHER.get("kind", "exp")
SMOOTH_KW = (
    {"converge_frac": float(SMOOTHER.get("converge_frac", 0.99))}
    if SMOOTH_KIND == "exp"
    else {
        "mid_frac": float(SMOOTHER.get("logistic", {}).get("mid_frac", 0.5)),
        "steepness": (SMOOTHER.get("logistic", {}).get("steepness", None)),
    }
)

AGEB = CFG["age_bins"]
def _coerce_list(x):
    if isinstance(x, list):
        # flatten nested one-level lists if any
        flat = []
        for it in x:
            if isinstance(it, list):
                flat.extend(it)
            elif isinstance(it, str) and (";" in it or "," in it):
                flat.extend([s.strip() for s in it.replace(",", ";").split(";") if s.strip()])
            else:
                flat.append(it)
        return flat
    if isinstance(x, str):
        if ";" in x or "," in x:
            return [s.strip() for s in x.replace(",", ";").split(";") if s.strip()]
        return [x.strip()]
    # fallback to defaults if malformed
    return _DEFAULT_CFG["age_bins"]["expected_bins"]

EXPECTED_BINS = _coerce_list(AGEB.get("expected_bins", _DEFAULT_CFG["age_bins"]["expected_bins"]))
EDAD_ORDER = _coerce_list(AGEB.get("order", _DEFAULT_CFG["age_bins"]["order"]))

RUNS = CFG["runs"]
MODE = RUNS.get("mode", "no_draws")
NO_DRAWS_TASKS = RUNS.get("no_draws_tasks", _DEFAULT_CFG["runs"]["no_draws_tasks"])
DRAWS = RUNS.get("draws", _DEFAULT_CFG["runs"]["draws"])

N_PROCS = int(CFG.get("parallel", {}).get("processes", 1)) or 1

FILENAMES = CFG["filenames"]


# ------------------------------ Helper functions ------------------------------

def get_target_tfrs(file_path: str) -> dict:
    df = pd.read_csv(file_path)
    return dict(zip(df["DPTO_NOMBRE"], df["Target_TFR"]))

def _with_suffix(fname: str, suffix: str) -> str:
    """Insert suffix before extension, if suffix is non-empty (e.g., 'asfr.csv' + '_draw1' -> 'asfr_draw1.csv')."""
    if not suffix:
        return fname
    base, ext = os.path.splitext(fname)
    return f"{base}{suffix}{ext}"

def _bin_width(label: str) -> float:
    s = str(label)
    if "-" in s:
        lo, hi = s.split("-")
        return float(hi) - float(lo) + 1.0
    if s.endswith("+"):
        return 5.0
    return 1.0

def _widths_from_index(idx) -> np.ndarray:
    return np.array([_bin_width(x) for x in idx], dtype=float)

def _tfr_from_asfr_df(asfr_df: pd.DataFrame) -> float:
    s = asfr_df["asfr"].astype(float)
    widths = _widths_from_index(s.index)
    return float(np.sum(s.values * widths))

def _normalize_weights_to(idx, w):
    """Reindex weights to `idx` and renormalize so sum(w*Δa)=1 over that index."""
    idx = pd.Index(idx).astype(str).str.strip()
    w = pd.Series(w, copy=False)
    w.index = w.index.astype(str).str.strip()
    w = w.reindex(idx).fillna(0.0)
    widths = _widths_from_index(idx)
    s = float(np.sum(w.values * widths))
    if not np.isfinite(s) or s <= 0:
        # uniform fallback
        w = pd.Series(1.0, index=idx) / len(idx)
        s = float(np.sum(w.values * widths))
    return w / s

def _exp_tfr(TFR0: float, target: float, years: int, step: int, converge_frac: float = 0.99) -> float:
    t = max(step, 0)
    gap0 = float(TFR0 - target)
    if gap0 == 0.0:
        return float(target)
    eps = 1e-12
    q = max(1.0 - converge_frac, eps)
    kappa = -np.log(q) / max(years, 1)
    return float(target + gap0 * np.exp(-kappa * t))

def _logistic_tfr(TFR0: float, target: float, years: int, step: int,
                  mid_frac: float = 0.5, steepness: float | None = None) -> float:
    t = max(step, 0.0)
    t0 = float(mid_frac) * max(years, 1)
    if steepness is None:
        r = 100.0
        steepness = (np.log(r) - np.log(1 + np.exp(-t0))) / max(years - t0, 1e-9)
    s = float(steepness)
    gap0 = float(TFR0 - target)
    scale = 1.0 + np.exp(-s * t0)
    return float(target + gap0 * (scale / (1.0 + np.exp(s * (t - t0)))))

def _smooth_tfr(TFR0: float, target: float, years: int, step: int, kind: str = "exp", **kwargs) -> float:
    if kind == "exp":
        return _exp_tfr(TFR0, target, years, step, **kwargs)
    elif kind == "logistic":
        return _logistic_tfr(TFR0, target, years, step, **kwargs)
    else:
        raise ValueError(f"Unknown TFR path kind: {kind!r}")

def fill_missing_age_bins(s: pd.Series) -> pd.Series:
    # Uses EXPECTED_BINS from config
    return s.reindex(EXPECTED_BINS, fill_value=0)


# ------------------------------- Main procedure -------------------------------

def main_wrapper(conteos, emi, imi, projection_range, sample_type, distribution=None, draw=None):
    DTPO_list = list(conteos["DPTO_NOMBRE"].unique()) + ["total_nacional"]
    suffix = f"_{draw}" if distribution is not None else ""

    asfr_weights = {}      # key: (DPTO, death_choice) -> w(a) with sum(w*Δa)=1
    asfr_baseline = {}     # key -> dict(year=y0, TFR0=TFR0)

    # last observed year per death source from config
    last_obs_year_by_death = dict(LAST_OBS_YEAR)

    # --- Target TFR loading/diagnostics
    target_tfrs = None
    target_csv_path = PATHS["target_tfr_csv"]
    if os.path.exists(target_csv_path):
        target_tfrs = get_target_tfrs(target_csv_path)

        def _finite(x):
            try:
                return np.isfinite(float(x))
            except Exception:
                return False

        if PRINT_TARGET_CSV:
            dptos_no_nat = [d for d in DTPO_list if d != "total_nacional"]
            missing = [d for d in dptos_no_nat if not (d in target_tfrs and _finite(target_tfrs[d]))]
            if len(missing) == 0:
                print("target_csv present: per-DPTO targets available for all DPTOs.")
            else:
                print(f"target_csv present: missing/invalid targets for {len(missing)} DPTO(s); defaulting to global target where needed.")
            if ("total_nacional" in target_tfrs) and _finite(target_tfrs["total_nacional"]):
                print("target_csv includes a finite total_nacional target.")
            else:
                print("target_csv present but no finite total_nacional target; aggregating national target from DPTO targets.")
    else:
        if PRINT_TARGET_CSV:
            print("no target_csv file, defaulting to global target")

    def _is_finite_number(x) -> bool:
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    def _dept_target_or_default(dpto_name: str) -> float:
        """DPTO-specific target if present and finite; else DEFAULT_TFR_TARGET from config."""
        if (target_tfrs is not None) and (dpto_name in target_tfrs) and _is_finite_number(target_tfrs[dpto_name]):
            return float(target_tfrs[dpto_name])
        return float(DEFAULT_TFR_TARGET)

    def _national_weighted_target(year: int, death_choice: str, asfr_age_index) -> float:
        """
        Exposure-weighted aggregation of DPTO targets for 'total_nacional' when its own target
        is NaN/missing. Weights are female exposures in ASFR ages for the relevant year/source.
        """
        dptos_no_nat = [d for d in DTPO_list if d != "total_nacional"]
        asfr_ages = pd.Index(asfr_age_index).astype(str)

        if (year == 2018) or (death_choice != "EEVV"):
            base = (conteos[(conteos["VARIABLE"] == "poblacion_total") &
                            (conteos["FUENTE"] == "censo_2018") &
                            (conteos["SEXO"] == 2) &
                            (conteos["ANO"] == 2018) &
                            (conteos["DPTO_NOMBRE"].isin(dptos_no_nat))].copy())
            exp_by_dpto = (base[base["EDAD"].isin(asfr_ages)]
                           .groupby("DPTO_NOMBRE")["VALOR_corrected"].sum())
        else:
            base = (proj_F[(proj_F["year"] == year) &
                           (proj_F["death_choice"] == death_choice) &
                           (proj_F["DPTO_NOMBRE"].isin(dptos_no_nat))].copy())
            exp_by_dpto = (base[base["EDAD"].isin(asfr_ages)]
                           .groupby("DPTO_NOMBRE")["VALOR_corrected"].sum())

        exp_by_dpto = exp_by_dpto.fillna(0.0)
        total_exp = float(exp_by_dpto.sum())
        if not np.isfinite(total_exp) or total_exp <= 0.0:
            t_list = [_dept_target_or_default(d) for d in dptos_no_nat]
            return float(np.mean(t_list)) if len(t_list) else float(DEFAULT_TFR_TARGET)

        num = 0.0
        for d in dptos_no_nat:
            Ei = float(exp_by_dpto.get(d, 0.0))
            wi = Ei / total_exp
            ti = _dept_target_or_default(d)
            num += wi * ti
        return float(num)

    def _target_for_this_unit(dpto_name: str, year: int, death_choice: str, asfr_age_index) -> float:
        if target_tfrs is None:
            return float(DEFAULT_TFR_TARGET)
        if dpto_name != "total_nacional":
            return _dept_target_or_default(dpto_name)
        if ("total_nacional" in target_tfrs) and _is_finite_number(target_tfrs["total_nacional"]):
            return float(target_tfrs["total_nacional"])
        return _national_weighted_target(year, death_choice, asfr_age_index)

    # Projection loop
    for death_choice in DEATH_CHOICES:
        proj_F = pd.DataFrame()
        proj_M = pd.DataFrame()
        proj_T = pd.DataFrame()

        for year in tqdm(projection_range):
            for DPTO in DTPO_list:
                if DPTO != "total_nacional":
                    conteos_all = conteos[conteos["DPTO_NOMBRE"] == DPTO]
                else:
                    conteos_all = conteos[conteos["DPTO_NOMBRE"] != DPTO]

                conteos_all_M = conteos_all[conteos_all["SEXO"] == 1.0]
                conteos_all_F = conteos_all[conteos_all["SEXO"] == 2.0]

                if (death_choice == "censo_2018"):
                    conteos_all_M = conteos_all_M[conteos_all_M["ANO"] == 2018]
                    conteos_all_M_d = conteos_all_M[(conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "censo_2018")]
                    conteos_all_F = conteos_all_F[conteos_all_F["ANO"] == 2018]
                    conteos_all_F_d = conteos_all_F[(conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "censo_2018")]
                elif death_choice == "EEVV":
                    if year > LAST_OBS_YEAR["EEVV"]:
                        yref = LAST_OBS_YEAR["EEVV"]
                        conteos_all_M = conteos_all_M[conteos_all_M["ANO"] == yref]
                        conteos_all_M_d = conteos_all_M[(conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "EEVV")]
                        conteos_all_F = conteos_all_F[conteos_all_F["ANO"] == yref]
                        conteos_all_F_d = conteos_all_F[(conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "EEVV")]
                    else:
                        conteos_all_M = conteos_all_M[conteos_all_M["ANO"] == year]
                        conteos_all_M_d = conteos_all_M[(conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "EEVV")]
                        conteos_all_F = conteos_all_F[conteos_all_F["ANO"] == year]
                        conteos_all_F_d = conteos_all_F[(conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "EEVV")]
                elif death_choice == "midpoint":
                    conteos_all_M = conteos_all_M[conteos_all_M["ANO"] == 2018]
                    conteos_all_F = conteos_all_F[conteos_all_F["ANO"] == 2018]
                    conteos_all_M_d_1 = conteos_all_M[(conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "EEVV")]
                    conteos_all_F_d_1 = conteos_all_F[(conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "EEVV")]
                    conteos_all_M_d_2 = conteos_all_M[(conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "censo_2018")]
                    conteos_all_F_d_2 = conteos_all_F[(conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "censo_2018")]
                    conteos_all_M_d = pd.merge(conteos_all_M_d_1, conteos_all_M_d_2,
                                               on=["DPTO_NOMBRE","SEXO","EDAD","ANO","VARIABLE"], suffixes=("_EEVV","_censo"))
                    conteos_all_M_d["VALOR_corrected"] = 0.5 * (conteos_all_M_d["VALOR_corrected_EEVV"] + conteos_all_M_d["VALOR_corrected_censo"])
                    conteos_all_M_d = conteos_all_M_d[["DPTO_NOMBRE","SEXO","EDAD","ANO","VARIABLE","VALOR_corrected"]]
                    conteos_all_F_d = pd.merge(conteos_all_F_d_1, conteos_all_F_d_2,
                                               on=["DPTO_NOMBRE","SEXO","EDAD","ANO","VARIABLE"], suffixes=("_EEVV","_censo"))
                    conteos_all_F_d["VALOR_corrected"] = 0.5 * (conteos_all_F_d["VALOR_corrected_EEVV"] + conteos_all_F_d["VALOR_corrected_censo"])
                    conteos_all_F_d = conteos_all_F_d[["DPTO_NOMBRE","SEXO","EDAD","ANO","VARIABLE","VALOR_corrected"]]

                conteos_all_M_n = conteos_all_M[(conteos_all_M["VARIABLE"] == "nacimientos") & (conteos_all_M["FUENTE"] == "EEVV")]
                conteos_all_F_n = conteos_all_F[(conteos_all_F["VARIABLE"] == "nacimientos") & (conteos_all_F["FUENTE"] == "EEVV")]

                if (death_choice == "censo_2018") or (death_choice == "midpoint"):
                    conteos_all_M_p = conteos_all_M[(conteos_all_M["VARIABLE"] == "poblacion_total") & (conteos_all_M["FUENTE"] == "censo_2018")]
                    conteos_all_F_p = conteos_all_F[(conteos_all_F["VARIABLE"] == "poblacion_total") & (conteos_all_F["FUENTE"] == "censo_2018")]
                else:
                    if year == 2018:
                        conteos_all_M_p = conteos_all_M[(conteos_all_M["VARIABLE"] == "poblacion_total") & (conteos_all_M["FUENTE"] == "censo_2018")]
                        conteos_all_F_p = conteos_all_F[(conteos_all_F["VARIABLE"] == "poblacion_total") & (conteos_all_F["FUENTE"] == "censo_2018")]
                    elif (year > 2018 and year <= LAST_OBS_YEAR["EEVV"]):
                        # USE PROJECTIONS FOR POPULATION
                        conteos_all_F_p = proj_F[(proj_F["year"] == year) & (proj_F["DPTO_NOMBRE"] == DPTO) & (proj_F["death_choice"] == death_choice)]
                        conteos_all_M_p = proj_M[(proj_M["year"] == year) & (proj_M["DPTO_NOMBRE"] == DPTO) & (proj_M["death_choice"] == death_choice)]

                if year > 2018:
                    conteos_all_F_p_updated = proj_F[(proj_F["year"] == year) & (proj_F["DPTO_NOMBRE"] == DPTO) & (proj_F["death_choice"] == death_choice)]
                    conteos_all_M_p_updated = proj_M[(proj_M["year"] == year) & (proj_M["DPTO_NOMBRE"] == DPTO) & (proj_M["death_choice"] == death_choice)]

                if DPTO == "total_nacional":
                    conteos_all_M_n_t = conteos_all_M_n.groupby(["EDAD"])["VALOR_corrected"].sum()
                    conteos_all_F_n_t = conteos_all_F_n.groupby(["EDAD"])["VALOR_corrected"].sum()
                    conteos_all_M_d_t = conteos_all_M_d.groupby(["EDAD"])["VALOR_corrected"].sum()
                    conteos_all_F_d_t = conteos_all_F_d.groupby(["EDAD"])["VALOR_corrected"].sum()
                    conteos_all_M_p_t = conteos_all_M_p.groupby(["EDAD"])["VALOR_corrected"].sum()
                    conteos_all_F_p_t = conteos_all_F_p.groupby(["EDAD"])["VALOR_corrected"].sum()
                    if year > 2018:
                        conteos_all_F_p_t_updated = conteos_all_F_p_updated.groupby(["EDAD"])["VALOR_corrected"].sum()
                        conteos_all_M_p_t_updated = conteos_all_M_p_updated.groupby(["EDAD"])["VALOR_corrected"].sum()
                else:
                    conteos_all_M_n_t = conteos_all_M_n.set_index("EDAD")["VALOR_corrected"]
                    conteos_all_F_n_t = conteos_all_F_n.set_index("EDAD")["VALOR_corrected"]
                    conteos_all_M_d_t = conteos_all_M_d.set_index("EDAD")["VALOR_corrected"]
                    conteos_all_F_d_t = conteos_all_F_d.set_index("EDAD")["VALOR_corrected"]
                    conteos_all_M_p_t = conteos_all_M_p.set_index("EDAD")["VALOR_corrected"]
                    conteos_all_F_p_t = conteos_all_F_p.set_index("EDAD")["VALOR_corrected"]
                    if year > 2018:
                        conteos_all_M_p_t_updated = conteos_all_M_p_updated.set_index("EDAD")["VALOR_corrected"]
                        conteos_all_F_p_t_updated = conteos_all_F_p_updated.set_index("EDAD")["VALOR_corrected"]

                # --- ratio to align national totals
                edad_order = EDAD_ORDER

                if year == 2018:
                    lhs_M = (pd.Series(conteos_all_M_p_t).reindex(edad_order, fill_value=0).astype(float).rename("lhs"))
                    rhs_M = (conteos[(conteos["DPTO_NOMBRE"] != "total_nacional") & (conteos["SEXO"] == 1) &
                                     (conteos["ANO"] == 2018) & (conteos["VARIABLE"] == "poblacion_total") &
                                     (conteos["FUENTE"] == "censo_2018")]
                             .groupby("EDAD")["VALOR_corrected"].sum()
                             .reindex(edad_order, fill_value=0).astype(float).rename("rhs"))
                    ratio_M = lhs_M.div(rhs_M.replace(0, np.nan))

                    lhs_F = (pd.Series(conteos_all_F_p_t).reindex(edad_order, fill_value=0).astype(float).rename("lhs"))
                    rhs_F = (conteos[(conteos["DPTO_NOMBRE"] != "total_nacional") & (conteos["SEXO"] == 2) &
                                     (conteos["ANO"] == 2018) & (conteos["VARIABLE"] == "poblacion_total") &
                                     (conteos["FUENTE"] == "censo_2018")]
                             .groupby("EDAD")["VALOR_corrected"].sum()
                             .reindex(edad_order, fill_value=0).astype(float).rename("rhs"))
                    ratio_F = lhs_F.div(rhs_F.replace(0, np.nan))
                else:
                    lhs_M = (pd.Series(conteos_all_M_p_t).reindex(edad_order, fill_value=0).astype(float).rename("lhs"))
                    rhs_M = proj_M[(proj_M["year"] == year) & (proj_M["DPTO_NOMBRE"] == "total_nacional") &
                                   (proj_M["death_choice"] == death_choice)].set_index("EDAD")["VALOR_corrected"]
                    ratio_M = lhs_M.div(rhs_M.replace(0, np.nan))

                    lhs_F = (pd.Series(conteos_all_F_p_t).reindex(edad_order, fill_value=0).astype(float).rename("lhs"))
                    rhs_F = proj_F[(proj_F["year"] == year) & (proj_F["DPTO_NOMBRE"] == "total_nacional") &
                                   (proj_F["death_choice"] == death_choice)].set_index("EDAD")["VALOR_corrected"]
                    ratio_F = lhs_F.div(rhs_F.replace(0, np.nan))

                # --- migration: latest available single-year flows, scaled to 5-year period
                flows_year = min(year, FLOWS_LATEST_YEAR)
                imi_age_M = (imi.loc[(imi["ANO"] == flows_year) & (imi["SEXO"] == 1)]
                             .groupby("EDAD")["VALOR"].sum().reindex(edad_order, fill_value=0))
                emi_age_M = (emi.loc[(emi["ANO"] == flows_year) & (emi["SEXO"] == 1)]
                             .groupby("EDAD")["VALOR"].sum().reindex(edad_order, fill_value=0))
                net_M_annual = ratio_M * (imi_age_M - emi_age_M)
                net_M = PERIOD_YEARS * net_M_annual

                imi_age_F = (imi.loc[(imi["ANO"] == flows_year) & (imi["SEXO"] == 2)]
                             .groupby("EDAD")["VALOR"].sum().reindex(edad_order, fill_value=0))
                emi_age_F = (emi.loc[(emi["ANO"] == flows_year) & (emi["SEXO"] == 2)]
                             .groupby("EDAD")["VALOR"].sum().reindex(edad_order, fill_value=0))
                net_F_annual = ratio_F * (imi_age_F - emi_age_F)
                net_F = PERIOD_YEARS * net_F_annual

                # add half-period migration to exposures before rates; keep exposures > 0
                conteos_all_M_p_t = (conteos_all_M_p_t + (net_M / 2.0)).clip(lower=1e-9)
                conteos_all_F_p_t = (conteos_all_F_p_t + (net_F / 2.0)).clip(lower=1e-9)

                # --- Life tables (5-year block)
                if ((death_choice == "EEVV") and (year < LAST_OBS_YEAR["EEVV"] + 1)) or ((death_choice != "EEVV") and (year == 2018)):
                    lt_M_t = make_lifetable(fill_missing_age_bins(conteos_all_M_d_t).index,
                                            fill_missing_age_bins(conteos_all_M_p_t),
                                            fill_missing_age_bins(conteos_all_M_d_t))
                    lt_F_t = make_lifetable(fill_missing_age_bins(conteos_all_F_d_t).index,
                                            fill_missing_age_bins(conteos_all_F_p_t),
                                            fill_missing_age_bins(conteos_all_F_d_t))
                    lt_T_t = make_lifetable(fill_missing_age_bins(conteos_all_M_d_t).index,
                                            fill_missing_age_bins(conteos_all_M_p_t) + fill_missing_age_bins(conteos_all_F_p_t),
                                            fill_missing_age_bins(conteos_all_M_d_t) + fill_missing_age_bins(conteos_all_F_d_t))
                    # write LTs under configured results_dir
                    if distribution is None:
                        lt_path = os.path.join(PATHS["results_dir"], "lifetables", DPTO, sample_type, death_choice, str(year))
                    else:
                        lt_path = os.path.join(PATHS["results_dir"], "lifetables", DPTO, sample_type, death_choice, distribution, str(year))
                    os.makedirs(lt_path, exist_ok=True)
                    lt_M_t.to_csv(os.path.join(lt_path, _with_suffix(FILENAMES["lt_M"], suffix)))
                    lt_F_t.to_csv(os.path.join(lt_path, _with_suffix(FILENAMES["lt_F"], suffix)))
                    lt_T_t.to_csv(os.path.join(lt_path, _with_suffix(FILENAMES["lt_T"], suffix)))

                # --- ASFR (robust; fallbacks if TFR0 invalid)
                cutoff = last_obs_year_by_death[death_choice]
                key = (DPTO, death_choice)

                asfr_df = compute_asfr(
                    conteos_all_F_n_t.index,
                    pd.Series(conteos_all_F_p_t[conteos_all_F_p_t.index.isin(conteos_all_F_n_t.index)]),
                    pd.Series(conteos_all_F_n_t) + pd.Series(conteos_all_M_n_t)
                )
                asfr_df = asfr_df[["population","births","asfr"]].astype(float)

                if year <= cutoff:
                    TFR0 = _tfr_from_asfr_df(asfr_df)
                    if not np.isfinite(TFR0) or TFR0 <= 0.0:
                        # fallback: use last known weights for this DPTO or national, scaled to baseline TFR
                        if key in asfr_weights and key in asfr_baseline:
                            w_norm = _normalize_weights_to(asfr_df.index, asfr_weights[key])
                            TFR0 = float(asfr_baseline[key]["TFR0"])
                            asfr_df["asfr"] = (w_norm * TFR0).astype(float)
                        else:
                            nat_key = ("total_nacional", death_choice)
                            if nat_key in asfr_weights and nat_key in asfr_baseline:
                                w_norm = _normalize_weights_to(asfr_df.index, asfr_weights[nat_key])
                                TFR0 = float(asfr_baseline[nat_key]["TFR0"])
                                asfr_df["asfr"] = (w_norm * TFR0).astype(float)
                            else:
                                raise ValueError(f"No usable ASFR for {key} in year {year} and no prior weights available.")
                        TFR0 = _tfr_from_asfr_df(asfr_df)

                    # store weights/baseline from observed (or rescued) year
                    w = asfr_df["asfr"] / TFR0
                    asfr_weights[key] = w
                    asfr_baseline[key] = {"year": year, "TFR0": TFR0}
                    asfr = asfr_df
                else:
                    if key not in asfr_weights or key not in asfr_baseline:
                        raise KeyError(f"No baseline ASFR weights stored for {key}; did you process year {cutoff} first?")
                    w = asfr_weights[key]
                    base = asfr_baseline[key]
                    step = year - base["year"]

                    # DPTO-specific convergence target (or national weighted fallback)
                    TFR_TARGET_LOCAL = _target_for_this_unit(DPTO, year, death_choice, asfr_df.index)

                    TFR_t = _smooth_tfr(base["TFR0"], TFR_TARGET_LOCAL, CONV_YEARS, step, kind=SMOOTH_KIND, **SMOOTH_KW)

                    proj_df = asfr_df.copy()
                    proj_df["population"] = np.nan
                    proj_df["births"] = np.nan
                    w_norm = _normalize_weights_to(proj_df.index, w)
                    proj_df["asfr"] = (w_norm * TFR_t).astype(float)

                    widths = _widths_from_index(proj_df.index)
                    chk = float(np.sum(proj_df["asfr"].values * widths))
                    if not np.isfinite(chk) or abs(chk - TFR_t) > 1e-6:
                        raise AssertionError(f"Normalization failed for {key} year {year}: {chk} vs {TFR_t}")
                    asfr_df = proj_df
                    asfr = proj_df

                # write ASFR under configured results_dir
                if distribution is None:
                    asfr_path = os.path.join(PATHS["results_dir"], "asfr", DPTO, sample_type, str(year))
                else:
                    asfr_path = os.path.join(PATHS["results_dir"], "asfr", DPTO, sample_type, distribution, str(year))
                os.makedirs(asfr_path, exist_ok=True)
                asfr_df.to_csv(os.path.join(asfr_path, _with_suffix(FILENAMES["asfr"], suffix)))

                # --- Projections (Leslie step = 5 years; pass full 5-yr net migration)
                if year == 2018:
                    L_MM, L_MF, L_FF, age_structures_df_M, age_structures_df_F, age_structures_df_T = make_projections(
                        net_F, net_M, len(lt_F_t) - 1, 1, 2,
                        pd.Series(conteos_all_M_n_t), pd.Series(conteos_all_F_n_t),
                        lt_F_t, lt_M_t,
                        pd.Series(conteos_all_F_p_t), pd.Series(conteos_all_M_p_t),
                        asfr, 100000, year, DPTO, death_choice=death_choice
                    )
                else:
                    L_MM, L_MF, L_FF, age_structures_df_M, age_structures_df_F, age_structures_df_T = make_projections(
                        net_F, net_M, len(lt_F_t) - 1, 1, 2,
                        pd.Series(conteos_all_M_n_t), pd.Series(conteos_all_F_n_t),
                        lt_F_t, lt_M_t,
                        pd.Series(conteos_all_F_p_t_updated), pd.Series(conteos_all_M_p_t_updated),
                        asfr, 100000, year, DPTO, death_choice=death_choice
                    )
                save_LL(L_MM, L_MF, L_FF, death_choice, DPTO, sample_type, distribution, suffix, year)

                proj_F = pd.concat([proj_F, age_structures_df_F], axis=0, ignore_index=True, sort=False)
                proj_M = pd.concat([proj_M, age_structures_df_M], axis=0, ignore_index=True, sort=False)
                proj_T = pd.concat([proj_T, age_structures_df_T], axis=0, ignore_index=True, sort=False)

        save_projections(proj_F, proj_M, proj_T, sample_type, distribution, suffix, death_choice, year)


# ----------------------------------- main -------------------------------------

if __name__ == "__main__":
    # Construct projection range from config
    projection_range = range(START_YEAR, END_YEAR + 1, STEP_YEARS)

    data = load_all_data(PATHS["data_dir"])
    conteos = data["conteos"]

    # Migration tables
    emi = conteos[conteos["VARIABLE"] == "crt_visa_F_emigracion"]
    imi = conteos[conteos["VARIABLE"] == "crt_visa_F_inmigracion"]

    # Build task list from config or CLI override ("draws")
    tasks = []
    if len(sys.argv) > 1 and sys.argv[1] == "draws":
        print("We'll be running this with draws")
        num_draws = int(DRAWS.get("num_draws", 1000))
        dist_types = list(DRAWS.get("dist_types", ["uniform", "pert", "beta", "normal"]))
        label_pattern = str(DRAWS.get("label_pattern", "{dist}_draw_{i}"))
        for dist in dist_types:
            for i in range(num_draws):
                label = label_pattern.format(dist=dist, i=i)
                tasks.append(("draw", dist, label))
    else:
        print("We'll be running this without draws")
        for t in NO_DRAWS_TASKS:
            tasks.append((t["sample_type"], t["distribution"], t["label"]))

    def _execute_task(args):
        sample_type, dist, label = args
        seed = zlib.adler32(label.encode("utf8")) & 0xFFFFFFFF
        np.random.seed(seed)

        df = conteos.copy()
        df["VALOR_withmissing"] = df["VALOR"]
        df["VALOR_corrected"] = np.nan

        processed_subsets = []
        for var in ["defunciones", "nacimientos", "poblacion_total"]:
            mask = df["VARIABLE"] == var
            df_var = df.loc[mask].copy()
            df_var = allocate_and_drop_missing_age(df_var)
            df_var.loc[:, "VALOR_corrected"] = correct_valor_for_omission(
                df_var, sample_type, distribution=dist, valor_col="VALOR_withmissing"
            )
            processed_subsets.append(df_var)

        df = pd.concat(processed_subsets, axis=0, ignore_index=True)
        df = df[df["EDAD"].notna()].copy()

        if dist is None:
            main_wrapper(df, emi, imi, projection_range, label)
        else:
            main_wrapper(df, emi, imi, projection_range, "draw", dist, label)
        return label

    # Parallel execution
    n_workers = N_PROCS if N_PROCS > 0 else 1
    with Pool(n_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_execute_task, tasks), total=len(tasks), desc="Tasks"):
            pass
