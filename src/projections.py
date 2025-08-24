import os
import re
import numpy as np
import pandas as pd


def save_projections(proj_F, proj_M, proj_T, sample_type, distribution, suffix, death_choice, year):
    base_proj_path = os.path.join('..', 'results', 'projections')
    for df_struct, key in [(proj_F, 'age_structures_df_F'),
                           (proj_M, 'age_structures_df_M'),
                           (proj_T, 'age_structures_df_T')]:
        out_path = os.path.join(base_proj_path, key, sample_type)
        if distribution:
            out_path = os.path.join(out_path, distribution)
        os.makedirs(out_path, exist_ok=True)
        df_struct.to_csv(os.path.join(out_path, f'{key}{suffix}{death_choice}.csv'))


def save_LL(L_MM, L_MF, L_FF, death_choice, DPTO, sample_type, distribution, suffix, year):
    base_proj_path = os.path.join('..', 'results', 'projections', death_choice, DPTO)
    for label, mat in [('L_MM', L_MM), ('L_MF', L_MF), ('L_FF', L_FF)]:
        out_path = os.path.join(base_proj_path, sample_type)
        if distribution:
            out_path = os.path.join(out_path, distribution)
        os.makedirs(out_path, exist_ok=True)
        pd.DataFrame(mat).to_csv(os.path.join(out_path, f'{label}{suffix}{year}.csv'))


def _s_open_from_ex(step: float, e: float) -> float:
    if not np.isfinite(e) or e <= 0:
        return 0.0
    return float(np.clip(np.exp(-step / e), 0.0, 1.0))


def _hazard_from_survival(s: float, step: float) -> float:
    s = float(np.clip(s, 1e-12, 1.0))
    return -np.log(s) / step


def make_projections(
    net_F, net_M,
    n, X, fert_start_idx,
    conteos_all_2018_M_n_t,
    conteos_all_2018_F_n_t,
    lt_2018_F_t,
    lt_2018_M_t,
    conteos_all_2018_F_p_t,
    conteos_all_2018_M_p_t,
    asfr_2018,
    l0,
    year,
    DPTO,
    death_choice,
    *,
    mort_improv_F: float = 0.015,
    mort_improv_M: float = 0.015
):
    k = n + 1
    columns = [f"t+{i}" for i in range(X + 1)]

    L_FF = np.zeros((k, k))
    L_MF = np.zeros((k, k))
    L_MM = np.zeros((k, k))

    srb = float(np.nansum(conteos_all_2018_M_n_t) / np.nansum(conteos_all_2018_F_n_t))
    p_f = 1.0 / (1.0 + srb)
    p_m = srb / (1.0 + srb)

    if "n" in lt_2018_F_t.columns:
        step = float(lt_2018_F_t["n"].iloc[0])
    else:
        age_idx = np.asarray(lt_2018_F_t.index, dtype=float)
        step = float(age_idx[1] - age_idx[0])

    if "lx" not in lt_2018_F_t.columns or "lx" not in lt_2018_M_t.columns:
        raise ValueError("Life tables must include an 'lx' column.")
    lxf = lt_2018_F_t["lx"].to_numpy(dtype=float)
    lxm = lt_2018_M_t["lx"].to_numpy(dtype=float)
    if len(lxf) != k or len(lxm) != k:
        raise ValueError(f"'lx' length must equal n+1 = {k} for both sexes.")

    sF_base = np.ones(k, dtype=float)
    sM_base = np.ones(k, dtype=float)
    for i in range(1, k):
        sF_base[i] = lxf[i] / lxf[i - 1]
        sM_base[i] = lxm[i] / lxm[i - 1]

    eF = float(lt_2018_F_t["ex"].iloc[-1])
    eM = float(lt_2018_M_t["ex"].iloc[-1])
    sF_open_base = _s_open_from_ex(step, eF)
    sM_open_base = _s_open_from_ex(step, eM)

    mF_base = np.array([_hazard_from_survival(s, step) for s in sF_base], dtype=float)
    mM_base = np.array([_hazard_from_survival(s, step) for s in sM_base], dtype=float)
    mF_base[-1] = _hazard_from_survival(sF_open_base, step)
    mM_base[-1] = _hazard_from_survival(sM_open_base, step)

    if hasattr(asfr_2018, "columns"):
        asfr_series = asfr_2018["asfr"]
        asfr_index = asfr_2018.index
    else:
        asfr_series = asfr_2018
        asfr_index = asfr_2018.index

    def lower_age(lbl):
        m = re.search(r"\d+", str(lbl))
        if not m:
            raise ValueError(f"Cannot parse age label: {lbl!r}")
        return int(m.group(0))

    fert_ages = [lower_age(a) for a in asfr_index]
    n_fert = len(fert_ages)
    fert_cols = [fert_start_idx + j for j in range(n_fert)]
    if fert_cols[-1] >= k:
        raise ValueError("Fertility columns exceed matrix dimension; check fert_start_idx and ASFR length.")
    births_per_woman = step * asfr_series.to_numpy(dtype=float)

    n0_F = np.asarray(conteos_all_2018_F_p_t, dtype=float).copy()
    n0_M = np.asarray(conteos_all_2018_M_p_t, dtype=float).copy()
    if n0_F.shape[0] != k or n0_M.shape[0] != k:
        raise ValueError(f"Initial population vectors must have length n+1 = {k}.")

    n_proj_F = [n0_F]
    n_proj_M = [n0_M]

    for t in range(1, X + 1):
        mF_t = mF_base * (1.0 - float(mort_improv_F))**t
        mM_t = mM_base * (1.0 - float(mort_improv_M))**t
        sF_t = np.exp(-mF_t * step)
        sM_t = np.exp(-mM_t * step)

        L_FF.fill(0.0); L_MF.fill(0.0); L_MM.fill(0.0)

        for i in range(1, k):
            L_FF[i, i - 1] = sF_t[i]
            L_MM[i, i - 1] = sM_t[i]

        L_FF[-1, -1] = sF_t[-1]
        L_MM[-1, -1] = sM_t[-1]

        S0_t = sF_t[1] if k > 1 else 1.0
        for j, col in enumerate(fert_cols):
            L_FF[0, col] = p_f * S0_t * births_per_woman[j]
            L_MF[0, col] = p_m * S0_t * births_per_woman[j]

        # one 5-year step; net_F/net_M are 5-year totals; we already added half before
        n_next_F = (L_FF @ n_proj_F[-1]) + (net_F / 2)
        n_next_M = (L_MM @ n_proj_M[-1]) + (net_M / 2) + (L_MF @ n_proj_F[-1])
        n_proj_F.append(n_next_F)
        n_proj_M.append(n_next_M)

    try:
        full_ages = lt_2018_F_t.index.astype(int)
    except Exception:
        if "age" in lt_2018_F_t.columns:
            full_ages = lt_2018_F_t["age"].astype(int).to_numpy()
        else:
            raise

    age_structure_matrix_F = np.column_stack(n_proj_F)
    age_structures_df_F = pd.DataFrame(age_structure_matrix_F, index=full_ages, columns=columns)

    age_structure_matrix_M = np.column_stack(n_proj_M)
    age_structures_df_M = pd.DataFrame(age_structure_matrix_M, index=full_ages, columns=columns)

    age_structures_df_T = age_structures_df_F + age_structures_df_M

    mapper = {
        0: "0-4", 5: "5-9", 10: "10-14", 15: "15-19", 20: "20-24",
        25: "25-29", 30: "30-34", 35: "35-39", 40: "40-44", 45: "45-49",
        50: "50-54", 55: "55-59", 60: "60-64", 65: "65-69", 70: "70-74",
        75: "75-79", 80: "80+"
    }

    # Label results with year + 5 (step width)
    step_int = int(round(step))

    age_structures_df_F = age_structures_df_F.reset_index().rename(columns={"index": "age"})
    age_structures_df_F["year"] = year + step_int
    if "t+1" not in age_structures_df_F.columns:
        raise KeyError("Column 't+1' not found in female projection matrix.")
    age_structures_df_F = age_structures_df_F.rename(columns={"age": "EDAD", "t+1": "VALOR_corrected"})
    age_structures_df_F["EDAD"] = age_structures_df_F["EDAD"].map(mapper)
    age_structures_df_F["DPTO_NOMBRE"] = DPTO
    age_structures_df_F["death_choice"] = death_choice
    age_structures_df_F = age_structures_df_F[["EDAD", "DPTO_NOMBRE", "year", "VALOR_corrected", "death_choice"]]

    age_structures_df_M = age_structures_df_M.reset_index().rename(columns={"index": "age"})
    age_structures_df_M["year"] = year + step_int
    if "t+1" not in age_structures_df_M.columns:
        raise KeyError("Column 't+1' not found in male projection matrix.")
    age_structures_df_M = age_structures_df_M.rename(columns={"age": "EDAD", "t+1": "VALOR_corrected"})
    age_structures_df_M["EDAD"] = age_structures_df_M["EDAD"].map(mapper)
    age_structures_df_M["DPTO_NOMBRE"] = DPTO
    age_structures_df_M["death_choice"] = death_choice
    age_structures_df_M = age_structures_df_M[["EDAD", "DPTO_NOMBRE", "year", "VALOR_corrected", "death_choice"]]

    age_structures_df_T = age_structures_df_T.reset_index().rename(columns={"index": "age"})
    age_structures_df_T["year"] = year + step_int
    if "t+1" not in age_structures_df_T.columns:
        raise KeyError("Column 't+1' not found in total projection matrix.")
    age_structures_df_T = age_structures_df_T.rename(columns={"age": "EDAD", "t+1": "VALOR_corrected"})
    age_structures_df_T["EDAD"] = age_structures_df_T["EDAD"].map(mapper)
    age_structures_df_T["DPTO_NOMBRE"] = DPTO
    age_structures_df_T["death_choice"] = death_choice
    age_structures_df_T = age_structures_df_T[["EDAD", "DPTO_NOMBRE", "year", "VALOR_corrected"]]

    return L_MM, L_MF, L_FF, age_structures_df_M, age_structures_df_F, age_structures_df_T
