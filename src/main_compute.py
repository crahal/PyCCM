# src/main_compute.py
# ------------------------------------------------------------------------------
# Population projection pipeline (UNABRIDGED default).
# - Reads optional per-DPTO mortality improvements from CSV (percent and schedule)
# - Uses YAML as global defaults and CSV as per-DPTO overrides
# - Supports time-decaying mortality improvement applied multiplicatively to hazards
# - Fertility targets can be provided by CSV (per DPTO) with convergence years
# - Midpoint (EEVV vs Censo) weighting per DPTO is supported via CSV
#
# Functionality: same as prior versions, with fixed + fully wired mortality
# improvements that now correctly read convergence_years (and other fields)
# from the CSV and apply them in projections.
# ------------------------------------------------------------------------------

from __future__ import annotations
import os
import sys
import zlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from mortality import make_lifetable
from fertility import compute_asfr, get_target_params
from projections import make_projections, save_LL, save_projections
from data_loaders import (
    load_all_data,
    correct_valor_for_omission,
    allocate_and_drop_missing_age,
    _load_config,
    return_default_config,
)
from helpers import (
    _single_year_bins, _widths_from_index, _coerce_list,
    get_midpoint_weights, _with_suffix, _tfr_from_asfr_df, _normalize_weights_to,
    _smooth_tfr, fill_missing_age_bins, _ridx, _collapse_defunciones_01_24_to_04,
)
from abridger import (
    unabridge_all, save_unabridged,
    harmonize_migration_to_90plus, harmonize_conteos_to_90plus,
)

# ------------------------------- Config loading -------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")
CFG, PATHS = _load_config(ROOT_DIR, CONFIG_PATH)

# Diagnostics
PRINT_TARGET_CSV = bool(CFG.get("diagnostics", {}).get("print_target_csv", True))
DEBUG_MORT = bool(CFG.get("diagnostics", {}).get("mortality_improvements_debug", False))

# Projections config
PROJ = CFG["projections"]
START_YEAR = int(PROJ["start_year"])
END_YEAR = int(PROJ["end_year"])
DEATH_CHOICES = list(PROJ["death_choices"])
LAST_OBS_YEAR = dict(PROJ["last_observed_year_by_death"])
FLOWS_LATEST_YEAR = int(PROJ["flows_latest_year"])

# Fertility config
FERT = CFG["fertility"]
DEFAULT_TFR_TARGET = float(FERT["default_tfr_target"])
CONV_YEARS = int(FERT["convergence_years"])
SMOOTHER = FERT["smoother"]
SMOOTH_KIND = SMOOTHER.get("kind", "exp")
SMOOTH_KW = (
    {"converge_frac": float(SMOOTHER.get("converge_frac", 0.99))}
    if SMOOTH_KIND == "exp"
    else {
        "mid_frac": float(SMOOTHER["logistic"].get("mid_frac", 0.5)),
        "steepness": (SMOOTHER["logistic"].get("steepness", None)),
    }
)

# Mortality base config (YAML defaults)
MORT_YAML = CFG.get("mortality", {})
MORT_USE_MA_DEFAULT = bool(MORT_YAML.get("use_ma", True))
MORT_MA_WINDOW_DEFAULT = int(MORT_YAML.get("ma_window", 5))
MORT_IMPROV_TOTAL_DEFAULT = float(MORT_YAML.get("improvement_total", 0.10))  # 10% long-run reduction
MORT_CONV_YEARS_DEFAULT = int(MORT_YAML.get("convergence_years", 50))
MORT_SMOOTHER_DEFAULT = MORT_YAML.get("smoother", {"kind": "exp", "converge_frac": 0.99})

# Midpoint weight default
DEFAULT_MIDPOINT = float(CFG.get("midpoints", {}).get("default_eevv_weight", 0.5))

# Filenames
FILENAMES = CFG.get("filenames", {})
LT_NAME_M = FILENAMES.get("lt_M", "lt_M_t.csv")
LT_NAME_F = FILENAMES.get("lt_F", "lt_F_t.csv")
LT_NAME_T = FILENAMES.get("lt_T", "lt_T_t.csv")

# ------------------------------ Age scaffolding ------------------------------
UNABR = bool(CFG.get("unabridging", {}).get("enabled", True))
if UNABR:
    EXPECTED_BINS = _single_year_bins()
    EDAD_ORDER = EXPECTED_BINS[:]
    STEP_YEARS = 1
    PERIOD_YEARS = 1
    print("[pipeline] UNABRIDGED mode: single-year ages & annual projections.")
else:
    AGEB = CFG["age_bins"]
    _exp_bins = _coerce_list(AGEB.get("expected_bins", return_default_config()["age_bins"]["expected_bins"]))
    _order = _coerce_list(AGEB.get("order", return_default_config()["age_bins"]["order"]))
    EXPECTED_BINS = _exp_bins if _exp_bins is not None else return_default_config()["age_bins"]["expected_bins"]
    EDAD_ORDER = _order if _order is not None else return_default_config()["age_bins"]["order"]
    STEP_YEARS = int(PROJ.get("step_years", 5)) or 5
    PERIOD_YEARS = STEP_YEARS
    print(f"[pipeline] ABRIDGED mode: 5-year ages & projections every {STEP_YEARS} years.")

# ---------------- Mortality improvements: CSV reader & parameter merge ----------------

def _coerce_percent_any(x):
    """
    Convert arbitrary cell to fractional rate in [0, 1):
      - '12', '12.0', 12 -> 0.12
      - '12%', '12.0 %'  -> 0.12
      - 0.12             -> 0.12   (already fractional)
    Returns None if not parseable or negative.
    """
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        if s.endswith("%"):
            v = float(s.strip("%").strip()) / 100.0
        else:
            v = float(s)
            if v > 1.0:
                v = v / 100.0
        if v < 0:
            return None
        return float(min(v, 0.999999))
    except Exception:
        return None

def _coerce_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None

def _coerce_int_pos(x):
    try:
        v = int(float(x))
        return v if v > 0 else None
    except Exception:
        return None

def _read_mortality_improvements_csv(path_csv: str | None) -> dict[str, dict]:
    """
    Read per-DPTO mortality improvement parameters from CSV.

    Required column:
      - DPTO_NOMBRE  (case-insensitive)

    Optional columns (any missing -> fallback to YAML defaults):
      - improvement_total  (alias: improvement, percent, pct)  % or fraction
      - convergence_years  integer > 0
      - kind               'exp' or 'logistic'
      - converge_frac      (0,1) for 'exp'
      - mid_frac           typically (0,1) for 'logistic'
      - steepness          float for 'logistic'

    Returns dict: { DPTO_NOMBRE: { possibly a subset of the above keys }, ... }
    """
    params_by_dpto: dict[str, dict] = {}
    csv_path = path_csv or PATHS.get("mortality_improvements_csv")
    if not csv_path or not os.path.exists(csv_path):
        if DEBUG_MORT:
            print(f"[mortality] No mortality_improvements CSV found (expected at {csv_path}). Using YAML defaults only.")
        return params_by_dpto

    df = pd.read_csv(csv_path)
    if df.empty:
        if DEBUG_MORT:
            print(f"[mortality] mortality_improvements CSV is empty: {csv_path}. Using YAML defaults only.")
        return params_by_dpto

    l2o = {c.lower(): c for c in df.columns}
    if "dpto_nombre" not in l2o:
        raise ValueError(f"[mortality] CSV must include DPTO_NOMBRE column: {csv_path}")
    dpto_col = l2o["dpto_nombre"]

    # improvement col: be flexible
    imp_candidates = ["improvement_total", "improvement", "percent", "pct"]
    imp_col = None
    for c in imp_candidates:
        if c in l2o:
            imp_col = l2o[c]
            break
    if imp_col is None:
        # Heuristic: first numeric-ish column other than DPTO
        for col in df.columns:
            if col == dpto_col:
                continue
            samp = df[col].head(8).tolist()
            if any(_coerce_percent_any(v) is not None for v in samp):
                imp_col = col
                break

    conv_col   = l2o.get("convergence_years")
    kind_col   = l2o.get("kind")
    cfrac_col  = l2o.get("converge_frac")
    mid_col    = l2o.get("mid_frac")
    steep_col  = l2o.get("steepness")

    for _, row in df.iterrows():
        dpto = str(row[dpto_col]).strip()
        if not dpto:
            continue
        rec: dict = {}
        # improvement_total
        if imp_col is not None and imp_col in row:
            imp = _coerce_percent_any(row.get(imp_col))
            if imp is not None:
                rec["improvement_total"] = imp
        # optional overrides
        if conv_col is not None and conv_col in row:
            cy = _coerce_int_pos(row.get(conv_col))
            if cy is not None:
                rec["convergence_years"] = cy
        if kind_col is not None and kind_col in row:
            k = row.get(kind_col)
            if isinstance(k, str) and k.strip():
                rec["kind"] = k.strip().lower()
        if cfrac_col is not None and cfrac_col in row:
            cf = _coerce_float(row.get(cfrac_col))
            if cf is not None:
                rec["converge_frac"] = float(np.clip(cf, 1e-6, 0.999999))
        if mid_col is not None and mid_col in row:
            mf = _coerce_float(row.get(mid_col))
            if mf is not None:
                rec["mid_frac"] = mf
        if steep_col is not None and steep_col in row:
            st = _coerce_float(row.get(steep_col))
            if st is not None:
                rec["steepness"] = st

        if rec:
            params_by_dpto[dpto] = rec

    if DEBUG_MORT:
        print(f"[mortality] Loaded {len(params_by_dpto)} DPTO rows from {os.path.basename(csv_path)}")
        try:
            cols_show = [dpto_col] + [c for c in [imp_col, conv_col, kind_col, cfrac_col, mid_col, steep_col] if c]
            print(df[cols_show].head(8).to_string(index=False))
        except Exception:
            pass

    return params_by_dpto

# Load CSV once at module load (OK; small)
MORT_PARAMS_BY_DPTO = _read_mortality_improvements_csv(PATHS.get("mortality_improvements_csv"))

def _params_for_dpto(dpto_name: str) -> dict:
    """
    Merge CSV-provided fields (when present) with YAML defaults.
    """
    p = MORT_PARAMS_BY_DPTO.get(dpto_name, {})
    # YAML defaults + CSV overrides
    out = {
        "improvement_total": p.get("improvement_total", MORT_IMPROV_TOTAL_DEFAULT),
        "convergence_years": p.get("convergence_years", MORT_CONV_YEARS_DEFAULT),
        "kind": p.get("kind", (MORT_SMOOTHER_DEFAULT or {}).get("kind", "exp")),
        "converge_frac": p.get("converge_frac", (MORT_SMOOTHER_DEFAULT or {}).get("converge_frac", 0.99)),
        "mid_frac": p.get("mid_frac", ((MORT_SMOOTHER_DEFAULT or {}).get("logistic", {}) or {}).get("mid_frac", 0.5)),
        "steepness": p.get("steepness", ((MORT_SMOOTHER_DEFAULT or {}).get("logistic", {}) or {}).get("steepness", None)),
        "use_ma": p.get("use_ma", MORT_USE_MA_DEFAULT),
        "ma_window": p.get("ma_window", MORT_MA_WINDOW_DEFAULT),
    }
    return out

def _mortality_factor_for_year(year: int, dpto_name: str) -> float:
    """
    Return multiplicative hazard factor m_t / m_base ∈ (0,1], with m_base the
    'base' hazard from the life table supplied to this year's projection call.

    The factor decays over time up to a total improvement 'improvement_total'
    over 'convergence_years', using either an exponential or logistic schedule.

    Let G = -log(1 - total_improvement) > 0. The 'effective improvement'
    applied by year t (years since START_YEAR) is:
        effective(t) = G * S(t)
    where S(t) ∈ [0,1) is the chosen schedule. The hazard factor is exp(-effective).
    """
    par = _params_for_dpto(dpto_name)
    if DEBUG_MORT and year in (START_YEAR, START_YEAR + 1, START_YEAR + 5, START_YEAR + 10, END_YEAR):
        # Will print once per several milestone years (per DPTO) during run
        pass

    t = max(0, int(year) - int(START_YEAR))
    if t <= 0:
        fac = 1.0
    else:
        total = float(np.clip(par["improvement_total"], 0.0, 0.999999))
        if total <= 0.0:
            fac = 1.0
        else:
            G = -np.log(1.0 - total)
            conv = max(int(par["convergence_years"]), 1)
            kind = str(par["kind"]).lower() if par.get("kind") else "exp"
            if kind == "exp":
                converge_frac = float(np.clip(par.get("converge_frac", 0.99), 1e-6, 0.999999))
                kappa = -np.log(1.0 - converge_frac) / conv
                S_t = 1.0 - np.exp(-kappa * t)
            elif kind == "logistic":
                mid_frac = float(par.get("mid_frac", 0.5))
                steep = par.get("steepness", None)
                if steep is None:
                    target = 0.99
                    denom = max(conv * (1.0 - mid_frac), 1e-6)
                    steep = -np.log(1.0/target - 1.0) / denom
                s = float(steep)
                x = (t / conv) - mid_frac
                S_t = 1.0 / (1.0 + np.exp(-s * x))
            else:
                S_t = 0.0
            S_t = float(np.clip(S_t, 0.0, 1.0))
            effective = G * S_t
            fac = float(np.exp(-effective))

    if DEBUG_MORT:
        print(
            f"[mortality] year={year} (START={START_YEAR}) debug factors:\n"
            f"  DPTO= {dpto_name:>15s} factor={fac:.6f} (scalar={max(0.0, 1.0 - fac):.6f}) "
            f"params={{'improvement_total': {par.get('improvement_total')}, "
            f"'convergence_years': {par.get('convergence_years')}, 'kind': {par.get('kind')}, "
            f"'converge_frac': {par.get('converge_frac')}, 'mid_frac': {par.get('mid_frac')}, "
            f"'steepness': {par.get('steepness')}}}"
        )
    return fac

# --------------------------------- main logic ---------------------------------
def main_wrapper(conteos, emi, imi, projection_range, sample_type, distribution=None, draw=None):
    dptos_in_input = list(conteos["DPTO_NOMBRE"].unique())
    DTPO_list = dptos_in_input + ["total_nacional"]
    suffix = f"_{draw}" if distribution is not None else ""

    asfr_weights = {}
    asfr_baseline = {}

    # Target TFRs + per-DPTO convergence years
    target_tfrs = None
    target_conv_years = None
    target_csv_path = PATHS["target_tfr_csv"]
    if os.path.exists(target_csv_path):
        targets, conv_years = get_target_params(target_csv_path)
        target_tfrs = targets
        target_conv_years = conv_years

        def _finite(x):
            try:
                return np.isfinite(float(x))
            except Exception:
                return False

        if PRINT_TARGET_CSV:
            dptos_no_nat = [d for d in DTPO_list if d != "total_nacional"]
            missing = [d for d in dptos_no_nat if not (d in targets and _finite(targets[d]))]
            if len(missing) == 0:
                print("target_csv present: per-DPTO targets available for all DPTOs.")
            else:
                print(
                    f"target_csv present: missing/invalid targets for {len(missing)} DPTO(s); defaulting to global target where needed."
                )
            if ("total_nacional" in targets) and _finite(targets["total_nacional"]):
                print("target_csv includes a finite total_nacional target.")
            else:
                print("target_csv present but no finite total_nacional target; aggregating national target from DPTO targets.")
            if target_conv_years:
                print(
                    f"target_csv includes custom convergence years for {len(target_conv_years)} DPTO(s); YAML default will apply elsewhere."
                )
            else:
                print("target_csv includes no custom convergence years; YAML default will apply to all units.")
    else:
        if PRINT_TARGET_CSV:
            print("no target_csv file, defaulting to global target & YAML convergence years")

    # Midpoint (EEVV weight) per DPTO
    mid_csv = PATHS.get("midpoints_csv")
    midpoint_weights = {}
    if mid_csv and os.path.exists(mid_csv):
        try:
            midpoint_weights = get_midpoint_weights(mid_csv)
            print(
                f"midpoints_csv present: {len(midpoint_weights)} DPTO weights loaded; default {DEFAULT_MIDPOINT} used for others."
            )
        except Exception as e:
            print(f"Warning: failed to read midpoints_csv ({e}); default {DEFAULT_MIDPOINT} will be used for all DPTOs.")
            midpoint_weights = {}
    else:
        print(f"no midpoints_csv found; using default EEVV weight = {DEFAULT_MIDPOINT} for all DPTOs.")

    def _midpoint_for(dpto_name: str) -> float:
        try:
            return float(midpoint_weights.get(dpto_name, DEFAULT_MIDPOINT))
        except Exception:
            return DEFAULT_MIDPOINT

    def _is_finite_number(x) -> bool:
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    def _dept_target_or_default(dpto_name: str) -> float:
        if (target_tfrs is not None) and (dpto_name in target_tfrs) and _is_finite_number(target_tfrs[dpto_name]):
            return float(target_tfrs[dpto_name])
        return float(DEFAULT_TFR_TARGET)

    def _conv_years_for_unit(dpto_name: str) -> int:
        """Per-DPTO convergence years from fertility targets CSV, else YAML default."""
        if (target_conv_years is not None) and (dpto_name in target_conv_years):
            try:
                cy = int(target_conv_years[dpto_name]);  # noqa
                if cy > 0:
                    return cy
            except Exception:
                pass
        return CONV_YEARS

    def _national_weighted_target(year: int, death_choice: str, asfr_age_index, proj_F_local: pd.DataFrame) -> float:
        dptos_no_nat = [d for d in DTPO_list if d != "total_nacional"]
        asfr_ages = pd.Index(asfr_age_index).astype(str)
        if (year == START_YEAR) or (death_choice != "EEVV"):
            base = (
                conteos[
                    (conteos["VARIABLE"] == "poblacion_total")
                    & (conteos["FUENTE"] == "censo_2018")
                    & (conteos["SEXO"] == 2)
                    & (conteos["ANO"] == START_YEAR)
                    & (conteos["DPTO_NOMBRE"].isin(dptos_no_nat))
                ].copy()
            )
            exp_by_dpto = base[base["EDAD"].isin(asfr_ages)].groupby("DPTO_NOMBRE")["VALOR_corrected"].sum()
        else:
            base = (
                proj_F_local[
                    (proj_F_local["year"] == year)
                    & (proj_F_local["death_choice"] == death_choice)
                    & (proj_F_local["DPTO_NOMBRE"].isin(dptos_no_nat))
                ].copy()
            )
            exp_by_dpto = base[base["EDAD"].isin(asfr_ages)].groupby("DPTO_NOMBRE")["VALOR_corrected"].sum()

        exp_by_dpto = exp_by_dpto.fillna(0.0)
        total_exp = float(exp_by_dpto.sum())
        if not np.isfinite(total_exp) or total_exp <= 0.0:
            t_list = [_dept_target_or_default(d) for d in dptos_no_nat]
            return float(np.mean(t_list)) if len(t_list) else float(DEFAULT_TFR_TARGET)

        num = 0.0
        for d in dptos_no_nat:
            Ei = float(exp_by_dpto.get(d, 0.0)); wi = Ei / total_exp
            ti = _dept_target_or_default(d); num += wi * ti
        return float(num)

    def _target_for_this_unit(dpto_name: str, year: int, death_choice: str, asfr_age_index, proj_F_local) -> float:
        if target_tfrs is None:
            return float(DEFAULT_TFR_TARGET)
        if dpto_name != "total_nacional":
            return _dept_target_or_default(dpto_name)
        if ("total_nacional" in target_tfrs) and _is_finite_number(target_tfrs["total_nacional"]):
            return float(target_tfrs["total_nacional"])
        return _national_weighted_target(year, death_choice, asfr_age_index, proj_F_local)

    # Projection loop
    for death_choice in DEATH_CHOICES:
        proj_F = pd.DataFrame()
        proj_M = pd.DataFrame()
        proj_T = pd.DataFrame()
        lt_M_t = None
        lt_F_t = None
        lt_T_t = None

        for year in tqdm(range(START_YEAR, END_YEAR + 1, STEP_YEARS)):
            for DPTO in (dptos_in_input + ["total_nacional"]):
                # select unit
                if DPTO != "total_nacional":
                    conteos_all = conteos[conteos["DPTO_NOMBRE"] == DPTO]
                else:
                    conteos_all = conteos[conteos["DPTO_NOMBRE"] != DPTO]

                conteos_all_M = conteos_all[conteos_all["SEXO"] == 1.0]
                conteos_all_F = conteos_all[conteos_all["SEXO"] == 2.0]

                # ------------------- select deaths source by death_choice/year
                if death_choice == "censo_2018":
                    year_for_deaths = START_YEAR
                    conteos_all_M = conteos_all_M[conteos_all_M["ANO"] == START_YEAR]
                    conteos_all_F = conteos_all_F[conteos_all_F["ANO"] == START_YEAR]
                    conteos_all_M_d = conteos_all_M[
                        (conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "censo_2018")
                    ]
                    conteos_all_F_d = conteos_all_F[
                        (conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "censo_2018")
                    ]

                elif death_choice == "EEVV":
                    year_for_deaths = min(year, LAST_OBS_YEAR["EEVV"])
                    conteos_all_M = conteos_all_M[conteos_all_M["ANO"] == year_for_deaths]
                    conteos_all_F = conteos_all_F[conteos_all_F["ANO"] == year_for_deaths]
                    conteos_all_M_d = conteos_all_M[
                        (conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "EEVV")
                    ]
                    conteos_all_F_d = conteos_all_F[
                        (conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "EEVV")
                    ]

                elif death_choice == "midpoint":
                    year_for_deaths = START_YEAR
                    conteos_all_M = conteos_all_M[conteos_all_M["ANO"] == START_YEAR]
                    conteos_all_F = conteos_all_F[conteos_all_F["ANO"] == START_YEAR]

                    # Merge deaths from both sources at row (DPTO, EDAD, etc.) level
                    merge_keys = ["DPTO_NOMBRE", "SEXO", "EDAD", "ANO", "VARIABLE"]
                    M_d1 = conteos_all_M[(conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "EEVV")]
                    F_d1 = conteos_all_F[(conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "EEVV")]
                    M_d2 = conteos_all_M[(conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "censo_2018")]
                    F_d2 = conteos_all_F[(conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "censo_2018")]

                    conteos_all_M_d = pd.merge(M_d1, M_d2, on=merge_keys, suffixes=("_EEVV", "_censo"))
                    w_M = conteos_all_M_d["DPTO_NOMBRE"].map(_midpoint_for).astype(float).clip(0.0, 1.0).fillna(DEFAULT_MIDPOINT)
                    conteos_all_M_d["VALOR_corrected"] = (
                        w_M * conteos_all_M_d["VALOR_corrected_EEVV"] + (1.0 - w_M) * conteos_all_M_d["VALOR_corrected_censo"]
                    )
                    conteos_all_M_d = conteos_all_M_d[["DPTO_NOMBRE", "SEXO", "EDAD", "ANO", "VARIABLE", "VALOR_corrected"]]

                    conteos_all_F_d = pd.merge(F_d1, F_d2, on=merge_keys, suffixes=("_EEVV", "_censo"))
                    w_F = conteos_all_F_d["DPTO_NOMBRE"].map(_midpoint_for).astype(float).clip(0.0, 1.0).fillna(DEFAULT_MIDPOINT)
                    conteos_all_F_d["VALOR_corrected"] = (
                        w_F * conteos_all_F_d["VALOR_corrected_EEVV"] + (1.0 - w_F) * conteos_all_F_d["VALOR_corrected_censo"]
                    )
                    conteos_all_F_d = conteos_all_F_d[["DPTO_NOMBRE", "SEXO", "EDAD", "ANO", "VARIABLE", "VALOR_corrected"]]
                else:
                    raise ValueError(f"Unknown death_choice: {death_choice}")

                # births (EEVV)
                conteos_all_M_n = conteos_all_M[
                    (conteos_all_M["VARIABLE"] == "nacimientos") & (conteos_all_M["FUENTE"] == "EEVV")
                ]
                conteos_all_F_n = conteos_all_F[
                    (conteos_all_F["VARIABLE"] == "nacimientos") & (conteos_all_F["FUENTE"] == "EEVV")
                ]

                # exposures used to compute rates
                if year == START_YEAR:
                    conteos_all_M_p = conteos_all_M[
                        (conteos_all_M["VARIABLE"] == "poblacion_total") & (conteos_all_M["FUENTE"] == "censo_2018")
                    ]
                    conteos_all_F_p = conteos_all_F[
                        (conteos_all_F["VARIABLE"] == "poblacion_total") & (conteos_all_F["FUENTE"] == "censo_2018")
                    ]
                else:
                    # from 2019+, always use the last projected exposures
                    conteos_all_F_p = proj_F[
                        (proj_F["year"] == year) & (proj_F["DPTO_NOMBRE"] == DPTO) & (proj_F["death_choice"] == death_choice)
                    ]
                    conteos_all_M_p = proj_M[
                        (proj_M["year"] == year) & (proj_M["DPTO_NOMBRE"] == DPTO) & (proj_M["death_choice"] == death_choice)
                    ]

                # Aggregate by EDAD to avoid duplicate labels downstream
                conteos_all_M_n_t = conteos_all_M_n.groupby("EDAD")["VALOR_corrected"].sum()
                conteos_all_F_n_t = conteos_all_F_n.groupby("EDAD")["VALOR_corrected"].sum()
                conteos_all_M_d_t = conteos_all_M_d.groupby("EDAD")["VALOR_corrected"].sum()
                conteos_all_F_d_t = conteos_all_F_d.groupby("EDAD")["VALOR_corrected"].sum()
                conteos_all_M_p_t = conteos_all_M_p.groupby("EDAD")["VALOR_corrected"].sum()
                conteos_all_F_p_t = conteos_all_F_p.groupby("EDAD")["VALOR_corrected"].sum()

                if year > START_YEAR:
                    conteos_all_F_p_t_updated = conteos_all_F_p_t.copy()
                    conteos_all_M_p_t_updated = conteos_all_M_p_t.copy()

                # --- ratio to align national totals
                edad_order = EDAD_ORDER

                if year == START_YEAR:
                    lhs_M = _ridx(conteos_all_M_p_t, edad_order).rename("lhs")
                    rhs_M = (
                        conteos[
                            (conteos["DPTO_NOMBRE"] != "total_nacional")
                            & (conteos["SEXO"] == 1) & (conteos["ANO"] == START_YEAR)
                            & (conteos["VARIABLE"] == "poblacion_total") & (conteos["FUENTE"] == "censo_2018")
                        ]
                        .groupby("EDAD")["VALOR_corrected"].sum()
                        .reindex(edad_order, fill_value=0).astype(float).rename("rhs")
                    )
                    ratio_M = lhs_M.div(rhs_M.replace(0, np.nan))

                    lhs_F = _ridx(conteos_all_F_p_t, edad_order).rename("lhs")
                    rhs_F = (
                        conteos[
                            (conteos["DPTO_NOMBRE"] != "total_nacional")
                            & (conteos["SEXO"] == 2) & (conteos["ANO"] == START_YEAR)
                            & (conteos["VARIABLE"] == "poblacion_total") & (conteos["FUENTE"] == "censo_2018")
                        ]
                        .groupby("EDAD")["VALOR_corrected"].sum()
                        .reindex(edad_order, fill_value=0).astype(float).rename("rhs")
                    )
                    ratio_F = lhs_F.div(rhs_F.replace(0, np.nan))
                else:
                    lhs_M = _ridx(conteos_all_M_p_t, edad_order).rename("lhs")
                    rhs_M = proj_M[
                        (proj_M["year"] == year) & (proj_M["DPTO_NOMBRE"] == "total_nacional") & (proj_M["death_choice"] == death_choice)
                    ].set_index("EDAD")["VALOR_corrected"]
                    rhs_M = _ridx(rhs_M, edad_order).rename("rhs")
                    ratio_M = lhs_M.div(rhs_M.replace(0, np.nan))

                    lhs_F = _ridx(conteos_all_F_p_t, edad_order).rename("lhs")
                    rhs_F = proj_F[
                        (proj_F["year"] == year) & (proj_F["DPTO_NOMBRE"] == "total_nacional") & (proj_F["death_choice"] == death_choice)
                    ].set_index("EDAD")["VALOR_corrected"]
                    rhs_F = _ridx(rhs_F, edad_order).rename("rhs")
                    ratio_F = lhs_F.div(rhs_F.replace(0, np.nan))

                ratio_M = ratio_M.replace([np.inf, -np.inf], np.nan).fillna(1.0)
                ratio_F = ratio_F.replace([np.inf, -np.inf], np.nan).fillna(1.0)

                # --- migration: latest available flows, scaled to the projection period
                flows_year = min(year, FLOWS_LATEST_YEAR)
                imi_age_M = (
                    imi.loc[(imi["ANO"] == flows_year) & (imi["SEXO"] == 1)].groupby("EDAD")["VALOR"].sum().reindex(edad_order, fill_value=0)
                )
                emi_age_M = (
                    emi.loc[(emi["ANO"] == flows_year) & (emi["SEXO"] == 1)].groupby("EDAD")["VALOR"].sum().reindex(edad_order, fill_value=0)
                )
                net_M_annual = ratio_M * (imi_age_M - emi_age_M)

                imi_age_F = (
                    imi.loc[(imi["ANO"] == flows_year) & (imi["SEXO"] == 2)].groupby("EDAD")["VALOR"].sum().reindex(edad_order, fill_value=0)
                )
                emi_age_F = (
                    emi.loc[(emi["ANO"] == flows_year) & (emi["SEXO"] == 2)].groupby("EDAD")["VALOR"].sum().reindex(edad_order, fill_value=0)
                )
                net_F_annual = ratio_F * (imi_age_F - emi_age_F)

                net_M_annual = net_M_annual.fillna(0.0)
                net_F_annual = net_F_annual.fillna(0.0)

                net_M = PERIOD_YEARS * net_M_annual
                net_F = PERIOD_YEARS * net_F_annual

                # add half-period migration to exposures before rates; keep exposures > 0
                conteos_all_M_p_t = (_ridx(conteos_all_M_p_t, edad_order) + (net_M / 2.0)).clip(lower=1e-9)
                conteos_all_F_p_t = (_ridx(conteos_all_F_p_t, edad_order) + (net_F / 2.0)).clip(lower=1e-9)
                if year > START_YEAR:
                    conteos_all_M_p_t_updated = (_ridx(conteos_all_M_p_t_updated, edad_order) + (net_M / 2.0)).clip(lower=1e-9)
                    conteos_all_F_p_t_updated = (_ridx(conteos_all_F_p_t_updated, edad_order) + (net_F / 2.0)).clip(lower=1e-9)

                # ------------------- Life tables: build & save if deaths exist for that source-year
                deaths_sum = float(conteos_all_M_d_t.sum() + conteos_all_F_d_t.sum())
                rebuild_lt_this_year = (
                    (death_choice == "EEVV" and (year <= LAST_OBS_YEAR["EEVV"]) and deaths_sum > 0.0)
                    or (death_choice in ("censo_2018", "midpoint") and year == START_YEAR and deaths_sum > 0.0)
                )

                # Resolve MA parameters possibly overridden per DPTO via CSV (rare)
                par_dp = _params_for_dpto(DPTO)
                USE_MA = bool(par_dp.get("use_ma", MORT_USE_MA_DEFAULT))
                MA_WIN = int(par_dp.get("ma_window", MORT_MA_WINDOW_DEFAULT))

                if rebuild_lt_this_year:
                    # choose exposures matching the death source-year context
                    if year > START_YEAR:
                        exp_M = fill_missing_age_bins(conteos_all_M_p_t_updated, edad_order)
                        exp_F = fill_missing_age_bins(conteos_all_F_p_t_updated, edad_order)
                    else:
                        exp_M = fill_missing_age_bins(conteos_all_M_p_t, edad_order)
                        exp_F = fill_missing_age_bins(conteos_all_F_p_t, edad_order)

                    lt_M_t = make_lifetable(
                        fill_missing_age_bins(conteos_all_M_d_t, edad_order).index,
                        exp_M,
                        fill_missing_age_bins(conteos_all_M_d_t, edad_order),
                        use_ma=USE_MA,
                        ma_window=MA_WIN,
                    )
                    lt_F_t = make_lifetable(
                        fill_missing_age_bins(conteos_all_F_d_t, edad_order).index,
                        exp_F,
                        fill_missing_age_bins(conteos_all_F_d_t, edad_order),
                        use_ma=USE_MA,
                        ma_window=MA_WIN,
                    )
                    lt_T_t = make_lifetable(
                        fill_missing_age_bins(conteos_all_M_d_t, edad_order).index,
                        exp_M + exp_F,
                        fill_missing_age_bins(conteos_all_M_d_t, edad_order)
                        + fill_missing_age_bins(conteos_all_F_d_t, edad_order),
                        use_ma=USE_MA,
                        ma_window=MA_WIN,
                    )
                    # write LTs
                    if lt_M_t is not None and lt_F_t is not None and lt_T_t is not None:
                        if distribution is None:
                            lt_path = os.path.join(
                                PATHS["results_dir"], "lifetables", DPTO, sample_type, death_choice, str(year_for_deaths)
                            )
                        else:
                            lt_path = os.path.join(
                                PATHS["results_dir"], "lifetables", DPTO, sample_type, death_choice, distribution, str(year_for_deaths)
                            )
                        os.makedirs(lt_path, exist_ok=True)
                        lt_M_t.to_csv(os.path.join(lt_path, _with_suffix(LT_NAME_M, f"_{draw}" if distribution else "")))
                        lt_F_t.to_csv(os.path.join(lt_path, _with_suffix(LT_NAME_F, f"_{draw}" if distribution else "")))
                        lt_T_t.to_csv(os.path.join(lt_path, _with_suffix(LT_NAME_T, f"_{draw}" if distribution else "")))

                # ------------------- ASFR
                cutoff = LAST_OBS_YEAR.get(death_choice, START_YEAR)
                key = (DPTO, death_choice)

                asfr_df = compute_asfr(
                    conteos_all_F_n_t.index,
                    pd.Series(conteos_all_F_p_t[conteos_all_F_p_t.index.isin(conteos_all_F_n_t.index)]),
                    pd.Series(conteos_all_F_n_t) + pd.Series(conteos_all_M_n_t),
                ).astype(float)

                if year <= cutoff:
                    TFR0 = _tfr_from_asfr_df(asfr_df)
                    if not np.isfinite(TFR0) or TFR0 <= 0.0:
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

                    conv_years_local = _conv_years_for_unit(DPTO)
                    TFR_TARGET_LOCAL = _target_for_this_unit(DPTO, year, death_choice, asfr_df.index, proj_F)

                    TFR_t = _smooth_tfr(
                        base["TFR0"], TFR_TARGET_LOCAL, conv_years_local, step, kind=SMOOTH_KIND, **SMOOTH_KW,
                    )

                    proj_df = asfr_df.copy()
                    proj_df["population"] = np.nan
                    proj_df["births"] = np.nan
                    w_norm = _normalize_weights_to(proj_df.index, w)
                    proj_df["asfr"] = (w_norm * TFR_t).astype(float)

                    widths = _widths_from_index(proj_df.index)
                    chk = float(np.sum(proj_df["asfr"].values * widths))
                    if not np.isfinite(chk) or abs(chk - TFR_t) > 1e-6:
                        raise AssertionError(f"Normalization failed for {key} year {year}: {chk} vs {TFR_t}")
                    asfr = proj_df

                # ------------------- Projection step (always project one step ahead)
                if lt_F_t is None or lt_M_t is None:
                    raise RuntimeError("Life tables not initialized before projection.")

                # Compute the time-decaying mortality improvement factor for this year & DPTO
                mort_factor = _mortality_factor_for_year(year, DPTO)
                mort_improv_scalar = float(np.clip(1.0 - mort_factor, 0.0, 0.999999))

                if year == START_YEAR:
                    L_MM, L_MF, L_FF, age_structures_df_M, age_structures_df_F, age_structures_df_T = make_projections(
                        net_F, net_M, len(lt_F_t) - 1, 1, 2,
                        _ridx(conteos_all_M_n_t, edad_order), _ridx(conteos_all_F_n_t, edad_order),
                        lt_F_t, lt_M_t,
                        _ridx(conteos_all_F_p_t, edad_order), _ridx(conteos_all_M_p_t, edad_order),
                        asfr, 100000, year, DPTO, death_choice=death_choice,
                        mort_improv_F=mort_improv_scalar, mort_improv_M=mort_improv_scalar,
                    )
                else:
                    L_MM, L_MF, L_FF, age_structures_df_M, age_structures_df_F, age_structures_df_T = make_projections(
                        net_F, net_M, len(lt_F_t) - 1, 1, 2,
                        _ridx(conteos_all_M_n_t, edad_order), _ridx(conteos_all_F_n_t, edad_order),
                        lt_F_t, lt_M_t,
                        _ridx(conteos_all_F_p_t_updated, edad_order), _ridx(conteos_all_M_p_t_updated, edad_order),
                        asfr, 100000, year, DPTO, death_choice=death_choice,
                        mort_improv_F=mort_improv_scalar, mort_improv_M=mort_improv_scalar,
                    )

                save_LL(
                    L_MM, L_MF, L_FF, death_choice, DPTO, sample_type, distribution, suffix, year, PATHS["results_dir"],
                )

                proj_F = pd.concat([proj_F, age_structures_df_F], axis=0, ignore_index=True, sort=False)
                proj_M = pd.concat([proj_M, age_structures_df_M], axis=0, ignore_index=True, sort=False)
                proj_T = pd.concat([proj_T, age_structures_df_T], axis=0, ignore_index=True, sort=False)

        # persist projections for this death_choice
        save_projections(
            proj_F, proj_M, proj_T, sample_type, distribution, suffix, death_choice, year, PATHS["results_dir"]
        )

# ----------------------------------- main -------------------------------------
if __name__ == "__main__":
    # dynamic step/period from UNABR
    projection_range = range(START_YEAR, END_YEAR + 1, STEP_YEARS)

    data = load_all_data(PATHS["data_dir"])
    conteos = data["conteos"]

    # Be liberal with migration label conventions
    mig_names_out = ["flujo_emigracion"]
    mig_names_in = ["flujo_inmigracion"]
    emi = conteos[conteos["VARIABLE"].isin(mig_names_out)].copy()
    imi = conteos[conteos["VARIABLE"].isin(mig_names_in)].copy()

    tasks = []
    mode = CFG.get("runs", {}).get("mode", "no_draws")
    if len(sys.argv) > 1 and sys.argv[1] == "draws":
        mode = "draws"

    if mode == "draws":
        print("We'll be running this with draws")
        DRAWS = CFG["runs"]["draws"]
        num_draws = int(DRAWS.get("num_draws", 1000))
        dist_types = list(DRAWS.get("dist_types", ["uniform", "pert", "beta", "normal"]))
        label_pattern = str(DRAWS.get("label_pattern", "{dist}_draw_{i}"))
        for dist in dist_types:
            for i in range(num_draws):
                label = label_pattern.format(dist=dist, i=i)
                tasks.append(("draw", dist, label))
    else:
        print("We'll be running this without draws")
        NO_DRAWS_TASKS = CFG["runs"]["no_draws_tasks"]
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

            if (not UNABR) and (var == "defunciones"):
                df_var = _collapse_defunciones_01_24_to_04(df_var)

            df_var = allocate_and_drop_missing_age(df_var)

            df_var.loc[:, "VALOR_corrected"] = correct_valor_for_omission(
                df_var, sample_type, distribution=dist, valor_col="VALOR_withmissing"
            )
            processed_subsets.append(df_var)

        df = pd.concat(processed_subsets, axis=0, ignore_index=True)
        df = df[df["EDAD"].notna()].copy()

        SERIES_KEYS = ["DPTO_NOMBRE", "DPTO_CODIGO", "ANO", "SEXO", "VARIABLE", "FUENTE", "OMISION"]

        if UNABR:
            # Tail harmonization BEFORE unabridging (pop & deaths to 90+; migration tails to 90+).
            df = harmonize_conteos_to_90plus(df, SERIES_KEYS, value_col="VALOR_corrected")
            pop_ref = df[df["VARIABLE"] == "poblacion_total"].copy()
            emi_90 = harmonize_migration_to_90plus(
                emi, pop_ref, SERIES_KEYS, value_col="VALOR", pop_value_col="VALOR_corrected"
            )
            imi_90 = harmonize_migration_to_90plus(
                imi, pop_ref, SERIES_KEYS, value_col="VALOR", pop_value_col="VALOR_corrected"
            )

            # Unabridge conteos (VALOR_corrected) and migration (VALOR) to 1-year EDADs
            unabridged = unabridge_all(
                df=df, emi=emi_90, imi=imi_90, series_keys=SERIES_KEYS, conteos_value_col="VALOR_corrected", ridge=1e-6,
            )
            save_unabridged(unabridged, os.path.join(PATHS["results_dir"], "unabridged"))

            conteos_in = unabridged["conteos"]
            emi_in = unabridged["emi"]
            imi_in = unabridged["imi"]
        else:
            # In 5-year mode we keep original bins and tails; no harmonization to 90+.
            conteos_in = df
            emi_in = emi.copy()
            imi_in = imi.copy()

        # Run projections with dynamic periodicity
        if dist is None:
            main_wrapper(conteos_in, emi_in, imi_in, projection_range, label)
        else:
            main_wrapper(conteos_in, emi_in, imi_in, projection_range, "draw", dist, label)
        return label

    n_workers = int(CFG.get("parallel", {}).get("processes", 1)) or 1
    with Pool(n_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_execute_task, tasks), total=len(tasks), desc="Tasks"):
            pass
