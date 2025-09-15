# src/main_compute.py

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
from data_loaders import (load_all_data,
                         correct_valor_for_omission,
                         allocate_and_drop_missing_age,
                         _load_config,
                         return_default_config
)
from helpers import (
    _single_year_bins, _bin_width, _widths_from_index, _coerce_list,
    get_midpoint_weights, _with_suffix, _tfr_from_asfr_df, _normalize_weights_to,
    _exp_tfr, _logistic_tfr, _smooth_tfr, fill_missing_age_bins, _ridx,
    _collapse_defunciones_01_24_to_04
)
from abridger import (
    unabridge_all, save_unabridged, SERIES_KEYS_DEFAULT,
    harmonize_migration_to_90plus, harmonize_conteos_to_90plus
)

# ------------------------------- Config loading -------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")
CFG, PATHS = _load_config(ROOT_DIR, CONFIG_PATH)
PRINT_TARGET_CSV = bool(CFG.get("diagnostics", {}).get("print_target_csv", True))
PROJ = CFG["projections"]
START_YEAR = int(PROJ["start_year"])
END_YEAR = int(PROJ["end_year"])
DEATH_CHOICES = list(PROJ["death_choices"])
LAST_OBS_YEAR = dict(PROJ["last_observed_year_by_death"])
FLOWS_LATEST_YEAR = int(PROJ["flows_latest_year"])
FERT = CFG["fertility"]
DEFAULT_TFR_TARGET = float(FERT["default_tfr_target"])
CONV_YEARS = int(FERT["convergence_years"])
SMOOTHER = FERT["smoother"]
SMOOTH_KIND = SMOOTHER.get("kind", "exp")
SMOOTH_KW = ({"converge_frac": float(SMOOTHER.get("converge_frac", 0.99))}
             if SMOOTH_KIND == "exp"
             else {"mid_frac": float(SMOOTHER["logistic"].get("mid_frac", 0.5)),
                   "steepness": (SMOOTHER["logistic"].get("steepness", None))})
DEFAULT_MIDPOINT = float(CFG.get("midpoints", {}).get("default_eevv_weight", 0.5))
FILENAMES = CFG.get("filenames", {})
LT_NAME_M = FILENAMES.get("lt_M", "lt_M_t.csv")
LT_NAME_F = FILENAMES.get("lt_F", "lt_F_t.csv")
LT_NAME_T = FILENAMES.get("lt_T", "lt_T_t.csv")

# ------------------------------ Age scaffolding ------------------------------
UNABR = bool(CFG.get("unabridging", {}).get("enabled", True))

if UNABR:
    EXPECTED_BINS = _single_year_bins()
    EDAD_ORDER    = EXPECTED_BINS[:]
    STEP_YEARS    = 1
    PERIOD_YEARS  = 1
    print("[pipeline] UNABRIDGED mode: single-year ages & annual projections.")
else:
    AGEB = CFG["age_bins"]
    _exp_bins = _coerce_list(AGEB.get("expected_bins", return_default_config()["age_bins"]["expected_bins"]))
    _order    = _coerce_list(AGEB.get("order",         return_default_config()["age_bins"]["order"]))
    EXPECTED_BINS = _exp_bins if _exp_bins is not None else return_default_config()["age_bins"]["expected_bins"]
    EDAD_ORDER    = _order    if _order    is not None else return_default_config()["age_bins"]["order"]
    STEP_YEARS    = int(PROJ.get("step_years", 5)) or 5
    PERIOD_YEARS  = STEP_YEARS
    print(f"[pipeline] ABRIDGED mode: 5-year ages & projections every {STEP_YEARS} years.")

# --------------------------------- main logic ---------------------------------

def main_wrapper(conteos, emi, imi, projection_range, sample_type, distribution=None, draw=None):
    DTPO_list = list(conteos["DPTO_NOMBRE"].unique()) + ["total_nacional"]
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
                print(f"target_csv present: missing/invalid targets for {len(missing)} DPTO(s); defaulting to global target where needed.")
            if ("total_nacional" in targets) and _finite(targets["total_nacional"]):
                print("target_csv includes a finite total_nacional target.")
            else:
                print("target_csv present but no finite total_nacional target; aggregating national target from DPTO targets.")

            if target_conv_years:
                print(f"target_csv includes custom convergence years for {len(target_conv_years)} DPTO(s); default (YAML) will apply elsewhere.")
            else:
                print("target_csv includes no custom convergence years; YAML default will apply to all units.")
    else:
        if PRINT_TARGET_CSV:
            print("no target_csv file, defaulting to global target & YAML convergence years")

    # Midpoint (EEVV weight) per DPTO
    mid_csv = PATHS["midpoints_csv"]
    midpoint_weights = {}
    if os.path.exists(mid_csv):
        try:
            midpoint_weights = get_midpoint_weights(mid_csv)
            print(f"midpoints_csv present: {len(midpoint_weights)} DPTO weights loaded; default {DEFAULT_MIDPOINT} used for others.")
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
        """Per-DPTO convergence years from CSV, else YAML default."""
        if (target_conv_years is not None) and (dpto_name in target_conv_years):
            try:
                cy = int(target_conv_years[dpto_name])
                if cy > 0:
                    return cy
            except Exception:
                pass
        return CONV_YEARS

    def _national_weighted_target(year: int, death_choice: str, asfr_age_index) -> float:
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
        proj_F = pd.DataFrame(); proj_M = pd.DataFrame(); proj_T = pd.DataFrame()
        lt_M_t = None; lt_F_t = None; lt_T_t = None

        for year in tqdm(range(START_YEAR, END_YEAR + 1, STEP_YEARS)):
            for DPTO in (list(conteos["DPTO_NOMBRE"].unique()) + ["total_nacional"]):
                if DPTO != "total_nacional":
                    conteos_all = conteos[conteos["DPTO_NOMBRE"] == DPTO]
                else:
                    conteos_all = conteos[conteos["DPTO_NOMBRE"] != DPTO]

                conteos_all_M = conteos_all[conteos_all["SEXO"] == 1.0]
                conteos_all_F = conteos_all[conteos_all["SEXO"] == 2.0]

                # ------------------- select deaths source by death_choice/year
                if death_choice == "censo_2018":
                    year_for_deaths = 2018
                    conteos_all_M = conteos_all_M[conteos_all_M["ANO"] == 2018]
                    conteos_all_F = conteos_all_F[conteos_all_F["ANO"] == 2018]
                    conteos_all_M_d = conteos_all_M[(conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "censo_2018")]
                    conteos_all_F_d = conteos_all_F[(conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "censo_2018")]

                elif death_choice == "EEVV":
                    year_for_deaths = min(year, LAST_OBS_YEAR["EEVV"])
                    conteos_all_M = conteos_all_M[conteos_all_M["ANO"] == year_for_deaths]
                    conteos_all_F = conteos_all_F[conteos_all_F["ANO"] == year_for_deaths]
                    conteos_all_M_d = conteos_all_M[(conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "EEVV")]
                    conteos_all_F_d = conteos_all_F[(conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "EEVV")]

                elif death_choice == "midpoint":
                    year_for_deaths = 2018
                    conteos_all_M = conteos_all_M[conteos_all_M["ANO"] == 2018]
                    conteos_all_F = conteos_all_F[conteos_all_F["ANO"] == 2018]

                    # Merge deaths from both sources at row (DPTO, EDAD, etc.) level
                    M_d1 = conteos_all_M[(conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "EEVV")]
                    F_d1 = conteos_all_F[(conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "EEVV")]
                    M_d2 = conteos_all_M[(conteos_all_M["VARIABLE"] == "defunciones") & (conteos_all_M["FUENTE"] == "censo_2018")]
                    F_d2 = conteos_all_F[(conteos_all_F["VARIABLE"] == "defunciones") & (conteos_all_F["FUENTE"] == "censo_2018")]

                    merge_keys = ["DPTO_NOMBRE","SEXO","EDAD","ANO","VARIABLE"]

                    conteos_all_M_d = pd.merge(M_d1, M_d2, on=merge_keys, suffixes=("_EEVV","_censo"))
                    w_M = conteos_all_M_d["DPTO_NOMBRE"].map(_midpoint_for).astype(float)
                    w_M = w_M.clip(0.0, 1.0).fillna(DEFAULT_MIDPOINT)
                    conteos_all_M_d["VALOR_corrected"] = (
                        w_M * conteos_all_M_d["VALOR_corrected_EEVV"]
                        + (1.0 - w_M) * conteos_all_M_d["VALOR_corrected_censo"]
                    )
                    conteos_all_M_d = conteos_all_M_d[["DPTO_NOMBRE","SEXO","EDAD","ANO","VARIABLE","VALOR_corrected"]]

                    conteos_all_F_d = pd.merge(F_d1, F_d2, on=merge_keys, suffixes=("_EEVV","_censo"))
                    w_F = conteos_all_F_d["DPTO_NOMBRE"].map(_midpoint_for).astype(float)
                    w_F = w_F.clip(0.0, 1.0).fillna(DEFAULT_MIDPOINT)
                    conteos_all_F_d["VALOR_corrected"] = (
                        w_F * conteos_all_F_d["VALOR_corrected_EEVV"]
                        + (1.0 - w_F) * conteos_all_F_d["VALOR_corrected_censo"]
                    )
                    conteos_all_F_d = conteos_all_F_d[["DPTO_NOMBRE","SEXO","EDAD","ANO","VARIABLE","VALOR_corrected"]]
                else:
                    raise ValueError(f"Unknown death_choice: {death_choice}")

                # births (EEVV)
                conteos_all_M_n = conteos_all_M[(conteos_all_M["VARIABLE"] == "nacimientos") & (conteos_all_M["FUENTE"] == "EEVV")]
                conteos_all_F_n = conteos_all_F[(conteos_all_F["VARIABLE"] == "nacimientos") & (conteos_all_F["FUENTE"] == "EEVV")]

                # population exposures used to compute rates
                if (death_choice in ("censo_2018", "midpoint")) or (year == 2018):
                    conteos_all_M_p = conteos_all_M[(conteos_all_M["VARIABLE"] == "poblacion_total") & (conteos_all_M["FUENTE"] == "censo_2018")]
                    conteos_all_F_p = conteos_all_F[(conteos_all_F["VARIABLE"] == "poblacion_total") & (conteos_all_F["FUENTE"] == "censo_2018")]
                else:
                    # for EEVV after 2018, use last computed projected exposures for this DPTO & year
                    conteos_all_F_p = proj_F[(proj_F["year"] == year) & (proj_F["DPTO_NOMBRE"] == DPTO) & (proj_F["death_choice"] == death_choice)]
                    conteos_all_M_p = proj_M[(proj_M["year"] == year) & (proj_M["DPTO_NOMBRE"] == DPTO) & (proj_M["death_choice"] == death_choice)]

                # convenience: aggregate by EDAD now (avoid duplicate labels downstream)
                conteos_all_M_n_t = conteos_all_M_n.groupby("EDAD")["VALOR_corrected"].sum()
                conteos_all_F_n_t = conteos_all_F_n.groupby("EDAD")["VALOR_corrected"].sum()
                conteos_all_M_d_t = conteos_all_M_d.groupby("EDAD")["VALOR_corrected"].sum()
                conteos_all_F_d_t = conteos_all_F_d.groupby("EDAD")["VALOR_corrected"].sum()
                conteos_all_M_p_t = conteos_all_M_p.groupby("EDAD")["VALOR_corrected"].sum()
                conteos_all_F_p_t = conteos_all_F_p.groupby("EDAD")["VALOR_corrected"].sum()

                if year > 2018 and death_choice == "EEVV":
                    conteos_all_F_p_t_updated = conteos_all_F_p_t.copy()
                    conteos_all_M_p_t_updated = conteos_all_M_p_t.copy()

                # --- ratio to align national totals
                edad_order = EDAD_ORDER

                if year == 2018:
                    lhs_M = _ridx(conteos_all_M_p_t, edad_order).rename("lhs")
                    rhs_M = (conteos[(conteos["DPTO_NOMBRE"] != "total_nacional") & (conteos["SEXO"] == 1) &
                                     (conteos["ANO"] == 2018) & (conteos["VARIABLE"] == "poblacion_total") &
                                     (conteos["FUENTE"] == "censo_2018")]
                             .groupby("EDAD")["VALOR_corrected"].sum()
                             .reindex(edad_order, fill_value=0).astype(float).rename("rhs"))
                    ratio_M = lhs_M.div(rhs_M.replace(0, np.nan))

                    lhs_F = _ridx(conteos_all_F_p_t, edad_order).rename("lhs")
                    rhs_F = (conteos[(conteos["DPTO_NOMBRE"] != "total_nacional") & (conteos["SEXO"] == 2) &
                                     (conteos["ANO"] == 2018) & (conteos["VARIABLE"] == "poblacion_total") &
                                     (conteos["FUENTE"] == "censo_2018")]
                             .groupby("EDAD")["VALOR_corrected"].sum()
                             .reindex(edad_order, fill_value=0).astype(float).rename("rhs"))
                    ratio_F = lhs_F.div(rhs_F.replace(0, np.nan))
                else:
                    lhs_M = _ridx(conteos_all_M_p_t, edad_order).rename("lhs")
                    rhs_M = proj_M[(proj_M["year"] == year) & (proj_M["DPTO_NOMBRE"] == "total_nacional") &
                                   (proj_M["death_choice"] == death_choice)].set_index("EDAD")["VALOR_corrected"]
                    rhs_M = _ridx(rhs_M, edad_order).rename("rhs")
                    ratio_M = lhs_M.div(rhs_M.replace(0, np.nan))

                    lhs_F = _ridx(conteos_all_F_p_t, edad_order).rename("lhs")
                    rhs_F = proj_F[(proj_F["year"] == year) & (proj_F["DPTO_NOMBRE"] == "total_nacional") &
                                   (proj_F["death_choice"] == death_choice)].set_index("EDAD")["VALOR_corrected"]
                    rhs_F = _ridx(rhs_F, edad_order).rename("rhs")
                    ratio_F = lhs_F.div(rhs_F.replace(0, np.nan))

                ratio_M = ratio_M.replace([np.inf, -np.inf], np.nan).fillna(1.0)
                ratio_F = ratio_F.replace([np.inf, -np.inf], np.nan).fillna(1.0)

                # --- migration: latest available flows, scaled to the projection period
                flows_year = min(year, FLOWS_LATEST_YEAR)
                imi_age_M = (imi.loc[(imi["ANO"] == flows_year) & (imi["SEXO"] == 1)]
                             .groupby("EDAD")["VALOR"].sum().reindex(edad_order, fill_value=0))
                emi_age_M = (emi.loc[(emi["ANO"] == flows_year) & (emi["SEXO"] == 1)]
                             .groupby("EDAD")["VALOR"].sum().reindex(edad_order, fill_value=0))
                net_M_annual = ratio_M * (imi_age_M - emi_age_M)

                imi_age_F = (imi.loc[(imi["ANO"] == flows_year) & (imi["SEXO"] == 2)]
                             .groupby("EDAD")["VALOR"].sum().reindex(edad_order, fill_value=0))
                emi_age_F = (emi.loc[(emi["ANO"] == flows_year) & (emi["SEXO"] == 2)]
                             .groupby("EDAD")["VALOR"].sum().reindex(edad_order, fill_value=0))
                net_F_annual = ratio_F * (imi_age_F - emi_age_F)

                net_M_annual = net_M_annual.fillna(0.0)
                net_F_annual = net_F_annual.fillna(0.0)

                net_M = PERIOD_YEARS * net_M_annual
                net_F = PERIOD_YEARS * net_F_annual

                # add half-period migration to exposures before rates; keep exposures > 0
                conteos_all_M_p_t = ( _ridx(conteos_all_M_p_t, edad_order) + (net_M / 2.0) ).clip(lower=1e-9)
                conteos_all_F_p_t = ( _ridx(conteos_all_F_p_t, edad_order) + (net_F / 2.0) ).clip(lower=1e-9)
                if year > 2018 and death_choice == "EEVV":
                    conteos_all_M_p_t_updated = ( _ridx(conteos_all_M_p_t_updated, edad_order) + (net_M / 2.0) ).clip(lower=1e-9)
                    conteos_all_F_p_t_updated = ( _ridx(conteos_all_F_p_t_updated, edad_order) + (net_F / 2.0) ).clip(lower=1e-9)

                # ------------------- Life tables: build & save IFF deaths exist for that source-year
                deaths_sum = float(conteos_all_M_d_t.sum() + conteos_all_F_d_t.sum())
                rebuild_lt_this_year = (
                    (death_choice == "EEVV" and (year <= LAST_OBS_YEAR["EEVV"]) and deaths_sum > 0.0) or
                    (death_choice in ("censo_2018", "midpoint") and year == 2018 and deaths_sum > 0.0)
                )

                if rebuild_lt_this_year:
                    # choose exposures matching the death source-year context
                    if death_choice == "EEVV" and year > 2018:
                        exp_M = fill_missing_age_bins(conteos_all_M_p_t_updated, edad_order)
                        exp_F = fill_missing_age_bins(conteos_all_F_p_t_updated, edad_order)
                    else:
                        exp_M = fill_missing_age_bins(conteos_all_M_p_t, edad_order)
                        exp_F = fill_missing_age_bins(conteos_all_F_p_t, edad_order)

                    lt_M_t = make_lifetable(
                        fill_missing_age_bins(conteos_all_M_d_t, edad_order).index,
                        exp_M,
                        fill_missing_age_bins(conteos_all_M_d_t, edad_order),
                        use_ma=CFG.get("mortality", {}).get("use_ma", True),
                        ma_window=int(CFG.get("mortality", {}).get("ma_window", 5)),
                    )
                    lt_F_t = make_lifetable(
                        fill_missing_age_bins(conteos_all_F_d_t, edad_order).index,
                        exp_F,
                        fill_missing_age_bins(conteos_all_F_d_t, edad_order),
                        use_ma=CFG.get("mortality", {}).get("use_ma", True),
                        ma_window=int(CFG.get("mortality", {}).get("ma_window", 5)),
                    )
                    lt_T_t = make_lifetable(
                        fill_missing_age_bins(conteos_all_M_d_t, edad_order).index,
                        exp_M + exp_F,
                        fill_missing_age_bins(conteos_all_M_d_t, edad_order) + fill_missing_age_bins(conteos_all_F_d_t, edad_order),
                        use_ma=CFG.get("mortality", {}).get("use_ma", True),
                        ma_window=int(CFG.get("mortality", {}).get("ma_window", 5)),
                    )
                    # write LTs
                    if lt_M_t is not None and lt_F_t is not None and lt_T_t is not None:
                        if distribution is None:
                            lt_path = os.path.join(PATHS["results_dir"], "lifetables", DPTO, sample_type, death_choice, str(year_for_deaths))
                        else:
                            lt_path = os.path.join(PATHS["results_dir"], "lifetables", DPTO, sample_type, death_choice, distribution, str(year_for_deaths))
                        os.makedirs(lt_path, exist_ok=True)
                        lt_M_t.to_csv(os.path.join(lt_path, _with_suffix(LT_NAME_M, f"_{draw}" if distribution else "")))
                        lt_F_t.to_csv(os.path.join(lt_path, _with_suffix(LT_NAME_F, f"_{draw}" if distribution else "")))
                        lt_T_t.to_csv(os.path.join(lt_path, _with_suffix(LT_NAME_T, f"_{draw}" if distribution else "")))

                # ------------------- ASFR
                cutoff = LAST_OBS_YEAR.get(death_choice, 2018)
                key = (DPTO, death_choice)

                asfr_df = compute_asfr(
                    conteos_all_F_n_t.index,
                    pd.Series(conteos_all_F_p_t[conteos_all_F_p_t.index.isin(conteos_all_F_n_t.index)]),
                    pd.Series(conteos_all_F_n_t) + pd.Series(conteos_all_M_n_t)
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

                    # per-DPTO convergence years
                    conv_years_local = _conv_years_for_unit(DPTO)

                    TFR_TARGET_LOCAL = _target_for_this_unit(DPTO, year, death_choice, asfr_df.index)
                    TFR_t = _smooth_tfr(
                        base["TFR0"], TFR_TARGET_LOCAL,
                        conv_years_local,
                        step,
                        kind=SMOOTH_KIND, **SMOOTH_KW
                    )

                    proj_df = asfr_df.copy()
                    proj_df["population"] = np.nan; proj_df["births"] = np.nan
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
                if year == 2018:
                    L_MM, L_MF, L_FF, age_structures_df_M, age_structures_df_F, age_structures_df_T = make_projections(
                        net_F, net_M, len(lt_F_t) - 1, 1, 2,
                        _ridx(conteos_all_M_n_t, edad_order), _ridx(conteos_all_F_n_t, edad_order),
                        lt_F_t, lt_M_t,
                        _ridx(conteos_all_F_p_t, edad_order), _ridx(conteos_all_M_p_t, edad_order),
                        asfr, 100000, year, DPTO, death_choice=death_choice
                    )
                else:
                    L_MM, L_MF, L_FF, age_structures_df_M, age_structures_df_F, age_structures_df_T = make_projections(
                        net_F, net_M, len(lt_F_t) - 1, 1, 2,
                        _ridx(conteos_all_M_n_t, edad_order), _ridx(conteos_all_F_n_t, edad_order),
                        lt_F_t, lt_M_t,
                        _ridx(conteos_all_F_p_t_updated if death_choice == "EEVV" else conteos_all_F_p_t, edad_order),
                        _ridx(conteos_all_M_p_t_updated if death_choice == "EEVV" else conteos_all_M_p_t, edad_order),
                        asfr, 100000, year, DPTO, death_choice=death_choice
                    )

                save_LL(L_MM, L_MF, L_FF, death_choice, DPTO, sample_type, distribution, suffix, year, PATHS["results_dir"])

                proj_F = pd.concat([proj_F, age_structures_df_F], axis=0, ignore_index=True, sort=False)
                proj_M = pd.concat([proj_M, age_structures_df_M], axis=0, ignore_index=True, sort=False)
                proj_T = pd.concat([proj_T, age_structures_df_T], axis=0, ignore_index=True, sort=False)

        # persist projections for this death_choice
        save_projections(proj_F, proj_M, proj_T, sample_type, distribution, suffix, death_choice, year, PATHS["results_dir"])

# ----------------------------------- main -------------------------------------

if __name__ == "__main__":
    # dynamic step/period from UNABR
    projection_range = range(START_YEAR, END_YEAR + 1, STEP_YEARS)

    data = load_all_data(PATHS["data_dir"])
    conteos = data["conteos"]

    # Be liberal with migration label conventions
    mig_names_out = ["flujo_emigracion"]
    mig_names_in  = ["flujo_inmigracion"]
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

        SERIES_KEYS = ["DPTO_NOMBRE","DPTO_CODIGO","ANO","SEXO","VARIABLE","FUENTE","OMISION"]

        if UNABR:
            # Tail harmonization BEFORE unabridging (pop & deaths to 90+; migration tails to 90+).
            df = harmonize_conteos_to_90plus(df, SERIES_KEYS, value_col="VALOR_corrected")
            pop_ref = df[df["VARIABLE"] == "poblacion_total"].copy()
            emi_90 = harmonize_migration_to_90plus(emi, pop_ref, SERIES_KEYS, value_col="VALOR", pop_value_col="VALOR_corrected")
            imi_90 = harmonize_migration_to_90plus(imi, pop_ref, SERIES_KEYS, value_col="VALOR", pop_value_col="VALOR_corrected")

            # Unabridge conteos (VALOR_corrected) and migration (VALOR) to 1-year EDADs
            unabridged = unabridge_all(
                df=df, emi=emi_90, imi=imi_90,
                series_keys=SERIES_KEYS,
                conteos_value_col="VALOR_corrected",
                ridge=1e-6,
            )
            save_unabridged(unabridged, os.path.join(PATHS["results_dir"], "unabridged"))

            conteos_in = unabridged["conteos"]
            emi_in     = unabridged["emi"]
            imi_in     = unabridged["imi"]
        else:
            # In 5-year mode we keep original bins and tails; no harmonization to 90+.
            conteos_in = df
            emi_in     = emi.copy()
            imi_in     = imi.copy()

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
