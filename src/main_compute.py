import pyreadr
import zlib
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from mortality import make_lifetable
from fertility import compute_asfr
from migration import create_migration_frame
from projections import make_projections, save_LL, save_projections
from data_loaders import load_all_data, correct_valor_for_omission, allocate_and_drop_missing_age


def _normalize_weights_to(idx, w):
    """Reindex weights to `idx` and renormalize so sum(w*Δa)=1 over that index."""
    idx = pd.Index(idx).astype(str).str.strip()
    w = pd.Series(w, copy=False)
    w.index = w.index.astype(str).str.strip()
    w = w.reindex(idx).fillna(0.0)
    widths = _widths_from_index(idx)
    s = float(np.sum(w.values * widths))
    if not np.isfinite(s) or s <= 0:
        # uniform fallback over the provided bins
        w = pd.Series(1.0, index=idx) / len(idx)
        s = float(np.sum(w.values * widths))
    return w / s


# --- smooth TFR paths ---------------------------------------------------------

def _exp_tfr(TFR0: float, target: float, years: int, step: int,
             converge_frac: float = 0.99) -> float:
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


def _smooth_tfr(TFR0: float, target: float, years: int, step: int,
                kind: str = "exp", **kwargs) -> float:
    if kind == "exp":
        return _exp_tfr(TFR0, target, years, step, **kwargs)
    elif kind == "logistic":
        return _logistic_tfr(TFR0, target, years, step, **kwargs)
    else:
        raise ValueError(f"Unknown TFR path kind: {kind!r}")


def _bin_width(label: str) -> float:
    s = str(label)
    if '-' in s:
        lo, hi = s.split('-')
        return float(hi) - float(lo) + 1.0
    if s.endswith('+'):
        return 5.0
    return 1.0

def _widths_from_index(idx) -> np.ndarray:
    return np.array([_bin_width(x) for x in idx], dtype=float)

def _tfr_from_asfr_df(asfr_df: pd.DataFrame) -> float:
    s = asfr_df['asfr'].astype(float)
    widths = _widths_from_index(s.index)
    return float(np.sum(s.values * widths))

def _linear_tfr(TFR0: float, target: float, years: int, step: int) -> float:
    u = min(max(step / years, 0.0), 1.0)
    return TFR0 + (target - TFR0) * u


def fill_missing_age_bins(s: pd.Series) -> pd.Series:
    expected_bins = [
        "0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
        "40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"
    ]
    return s.reindex(expected_bins, fill_value=0)


def main_wrapper(conteos, emi, imi, projection_range, sample_type, distribution=None, draw=None):
    DTPO_list = list(conteos['DPTO_NOMBRE'].unique()) + ['total_nacional']
    suffix = f"_{draw}" if distribution is not None else ''

    asfr_weights = {}      # key: (DPTO, death_choice) -> pd.Series w(a) with sum(w*Δa)=1
    asfr_baseline = {}     # key -> dict(year=y0, TFR0=TFR0)

    last_obs_year_by_death = {'EEVV': 2023, 'censo_2018': 2018, 'midpoint': 2018}

    TFR_TARGET = 1.5
    CONV_YEARS = 50
    PERIOD_YEARS = 5  # <<< 5-year block

    for death_choice in ['EEVV', 'censo_2018', 'midpoint']:
        proj_F = pd.DataFrame()
        proj_M = pd.DataFrame()
        proj_T = pd.DataFrame()

        for year in tqdm(projection_range):
            for DPTO in DTPO_list:
                if DPTO != 'total_nacional':
                    conteos_all = conteos[conteos['DPTO_NOMBRE'] == DPTO]
                else:
                    conteos_all = conteos[conteos['DPTO_NOMBRE'] != DPTO]

                conteos_all_M = conteos_all[conteos_all['SEXO'] == 1.0]
                conteos_all_F = conteos_all[conteos_all['SEXO'] == 2.0]

                if (death_choice == 'censo_2018'):
                    conteos_all_M = conteos_all_M[conteos_all_M['ANO'] == 2018]
                    conteos_all_M_d = conteos_all_M[(conteos_all_M['VARIABLE'] == 'defunciones') &
                                                    (conteos_all_M['FUENTE'] == 'censo_2018')]
                    conteos_all_F = conteos_all_F[conteos_all_F['ANO'] == 2018]
                    conteos_all_F_d = conteos_all_F[(conteos_all_F['VARIABLE'] == 'defunciones') &
                                                    (conteos_all_F['FUENTE'] == 'censo_2018')]
                elif death_choice == 'EEVV':
                    if year > 2023:
                        conteos_all_M = conteos_all_M[conteos_all_M['ANO'] == 2023]
                        conteos_all_M_d = conteos_all_M[(conteos_all_M['VARIABLE'] == 'defunciones') &
                                                        (conteos_all_M['FUENTE'] == 'EEVV')]
                        conteos_all_F = conteos_all_F[conteos_all_F['ANO'] == 2023]
                        conteos_all_F_d = conteos_all_F[(conteos_all_F['VARIABLE'] == 'defunciones') &
                                                        (conteos_all_F['FUENTE'] == 'EEVV')]
                    else:
                        conteos_all_M = conteos_all_M[conteos_all_M['ANO'] == year]
                        conteos_all_M_d = conteos_all_M[(conteos_all_M['VARIABLE'] == 'defunciones') &
                                                        (conteos_all_M['FUENTE'] == 'EEVV')]
                        conteos_all_F = conteos_all_F[conteos_all_F['ANO'] == year]
                        conteos_all_F_d = conteos_all_F[(conteos_all_F['VARIABLE'] == 'defunciones') &
                                                        (conteos_all_F['FUENTE'] == 'EEVV')]
                elif death_choice == 'midpoint':
                    conteos_all_M = conteos_all_M[conteos_all_M['ANO'] == 2018]
                    conteos_all_F = conteos_all_F[conteos_all_F['ANO'] == 2018]
                    conteos_all_M_d_1 = conteos_all_M[(conteos_all_M['VARIABLE'] == 'defunciones') &
                                                      (conteos_all_M['FUENTE'] == 'EEVV')]
                    conteos_all_F_d_1 = conteos_all_F[(conteos_all_F['VARIABLE'] == 'defunciones') &
                                                      (conteos_all_F['FUENTE'] == 'EEVV')]
                    conteos_all_M_d_2 = conteos_all_M[(conteos_all_M['VARIABLE'] == 'defunciones') &
                                                      (conteos_all_M['FUENTE'] == 'censo_2018')]
                    conteos_all_F_d_2 = conteos_all_F[(conteos_all_F['VARIABLE'] == 'defunciones') &
                                                      (conteos_all_F['FUENTE'] == 'censo_2018')]
                    conteos_all_M_d = pd.merge(conteos_all_M_d_1, conteos_all_M_d_2,
                                               on=['DPTO_NOMBRE', 'SEXO', 'EDAD', 'ANO', 'VARIABLE'],
                                               suffixes=('_EEVV', '_censo'))
                    conteos_all_M_d['VALOR_corrected'] = 0.5 * (conteos_all_M_d['VALOR_corrected_EEVV'] +
                                                                conteos_all_M_d['VALOR_corrected_censo'])
                    conteos_all_M_d = conteos_all_M_d[['DPTO_NOMBRE', 'SEXO', 'EDAD', 'ANO', 'VARIABLE',
                                                       'VALOR_corrected']]
                    conteos_all_F_d = pd.merge(conteos_all_F_d_1, conteos_all_F_d_2,
                                               on=['DPTO_NOMBRE', 'SEXO', 'EDAD', 'ANO', 'VARIABLE'],
                                               suffixes=('_EEVV', '_censo'))
                    conteos_all_F_d['VALOR_corrected'] = 0.5 * (conteos_all_F_d['VALOR_corrected_EEVV'] +
                                                                conteos_all_F_d['VALOR_corrected_censo'])
                    conteos_all_F_d = conteos_all_F_d[['DPTO_NOMBRE', 'SEXO', 'EDAD', 'ANO', 'VARIABLE',
                                                       'VALOR_corrected']]

                conteos_all_M_n = conteos_all_M[(conteos_all_M['VARIABLE'] == 'nacimientos') &
                                                (conteos_all_M['FUENTE'] == 'EEVV')]
                conteos_all_F_n = conteos_all_F[(conteos_all_F['VARIABLE'] == 'nacimientos') &
                                                (conteos_all_F['FUENTE'] == 'EEVV')]

                if (death_choice == 'censo_2018') or (death_choice == 'midpoint'):
                    conteos_all_M_p = conteos_all_M[(conteos_all_M['VARIABLE'] == 'poblacion_total') &
                                                    (conteos_all_M['FUENTE'] == 'censo_2018')]
                    conteos_all_F_p = conteos_all_F[(conteos_all_F['VARIABLE'] == 'poblacion_total') &
                                                    (conteos_all_F['FUENTE'] == 'censo_2018')]
                else:
                    if year == 2018:
                        conteos_all_M_p = conteos_all_M[(conteos_all_M['VARIABLE'] == 'poblacion_total') &
                                                        (conteos_all_M['FUENTE'] == 'censo_2018')]
                        conteos_all_F_p = conteos_all_F[(conteos_all_F['VARIABLE'] == 'poblacion_total') &
                                                        (conteos_all_F['FUENTE'] == 'censo_2018')]
                    elif (year > 2018 and year <= 2023):
                        # USE PROJECTIONS FOR POPULATION
                        conteos_all_F_p = proj_F[(proj_F['year'] == year) &
                            (proj_F['DPTO_NOMBRE'] == DPTO) &
                            (proj_F['death_choice'] == death_choice)]
                        conteos_all_M_p = proj_M[(proj_M['year'] == year) &
                            (proj_M['DPTO_NOMBRE'] == DPTO) &
                            (proj_M['death_choice'] == death_choice)]

                if year > 2018:
                    conteos_all_F_p_updated = proj_F[(proj_F['year'] == year) &
                        (proj_F['DPTO_NOMBRE'] == DPTO) &
                        (proj_F['death_choice'] == death_choice)]
                    conteos_all_M_p_updated = proj_M[(proj_M['year'] == year) &
                        (proj_M['DPTO_NOMBRE'] == DPTO) &
                        (proj_M['death_choice'] == death_choice)]

                if DPTO == 'total_nacional':
                    conteos_all_M_n_t = conteos_all_M_n.groupby(['EDAD'])['VALOR_corrected'].sum()
                    conteos_all_F_n_t = conteos_all_F_n.groupby(['EDAD'])['VALOR_corrected'].sum()
                    conteos_all_M_d_t = conteos_all_M_d.groupby(['EDAD'])['VALOR_corrected'].sum()
                    conteos_all_F_d_t = conteos_all_F_d.groupby(['EDAD'])['VALOR_corrected'].sum()
                    conteos_all_M_p_t = conteos_all_M_p.groupby(['EDAD'])['VALOR_corrected'].sum()
                    conteos_all_F_p_t = conteos_all_F_p.groupby(['EDAD'])['VALOR_corrected'].sum()
                    if year > 2018:
                        conteos_all_F_p_t_updated = conteos_all_F_p_updated.groupby(['EDAD'])['VALOR_corrected'].sum()
                        conteos_all_M_p_t_updated = conteos_all_M_p_updated.groupby(['EDAD'])['VALOR_corrected'].sum()
                else:
                    conteos_all_M_n_t = conteos_all_M_n.set_index('EDAD')['VALOR_corrected']
                    conteos_all_F_n_t = conteos_all_F_n.set_index('EDAD')['VALOR_corrected']
                    conteos_all_M_d_t = conteos_all_M_d.set_index('EDAD')['VALOR_corrected']
                    conteos_all_F_d_t = conteos_all_F_d.set_index('EDAD')['VALOR_corrected']
                    conteos_all_M_p_t = conteos_all_M_p.set_index('EDAD')['VALOR_corrected']
                    conteos_all_F_p_t = conteos_all_F_p.set_index('EDAD')['VALOR_corrected']
                    if year > 2018:
                        conteos_all_M_p_t_updated = conteos_all_M_p_updated.set_index('EDAD')['VALOR_corrected']
                        # FIX: use the proper source series here
                        conteos_all_F_p_t_updated = conteos_all_F_p_updated.set_index('EDAD')['VALOR_corrected']

                # --- ratio to align national totals
                edad_order = ['0-4','5-9','10-14','15-19','20-24','25-29',
                              '30-34','35-39','40-44','45-49','50-54','55-59',
                              '60-64','65-69','70-74','75-79','80+']

                if year == 2018:
                    lhs_M = (pd.Series(conteos_all_M_p_t).reindex(edad_order, fill_value=0).astype(float).rename('lhs'))
                    rhs_M = (conteos[(conteos['DPTO_NOMBRE'] != 'total_nacional') &
                                     (conteos['SEXO'] == 1) &
                                     (conteos['ANO'] == 2018) &
                                     (conteos['VARIABLE'] == 'poblacion_total') &
                                     (conteos['FUENTE'] == 'censo_2018')]
                             .groupby('EDAD')['VALOR_corrected'].sum()
                             .reindex(edad_order, fill_value=0).astype(float).rename('rhs'))
                    ratio_M = lhs_M.div(rhs_M.replace(0, np.nan))

                    lhs_F = (pd.Series(conteos_all_F_p_t).reindex(edad_order, fill_value=0).astype(float).rename('lhs'))
                    rhs_F = (conteos[(conteos['DPTO_NOMBRE'] != 'total_nacional') &
                                     (conteos['SEXO'] == 2) &
                                     (conteos['ANO'] == 2018) &
                                     (conteos['VARIABLE'] == 'poblacion_total') &
                                     (conteos['FUENTE'] == 'censo_2018')]
                             .groupby('EDAD')['VALOR_corrected'].sum()
                             .reindex(edad_order, fill_value=0).astype(float).rename('rhs'))
                    ratio_F = lhs_F.div(rhs_F.replace(0, np.nan))
                else:
                    # for 2023, 2028, ... use previous step's projection at the national level
                    lhs_M = (pd.Series(conteos_all_M_p_t).reindex(edad_order, fill_value=0).astype(float).rename('lhs'))
                    rhs_M = proj_M[(proj_M['year'] == year) &
                                   (proj_M['DPTO_NOMBRE'] == 'total_nacional') &
                                   (proj_M['death_choice'] == death_choice)].set_index('EDAD')['VALOR_corrected']
                    ratio_M = lhs_M.div(rhs_M.replace(0, np.nan))

                    lhs_F = (pd.Series(conteos_all_F_p_t).reindex(edad_order, fill_value=0).astype(float).rename('lhs'))
                    rhs_F = proj_F[(proj_F['year'] == year) &
                                   (proj_F['DPTO_NOMBRE'] == 'total_nacional') &
                                   (proj_F['death_choice'] == death_choice)].set_index('EDAD')['VALOR_corrected']
                    ratio_F = lhs_F.div(rhs_F.replace(0, np.nan))

                # --- migration: use latest available single-year flow, scale to 5-year period
                flows_year = min(year, 2021)  # keep your data cutoff logic
                imi_age_M = (imi.loc[(imi['ANO'] == flows_year) & (imi['SEXO'] == 1)]
                             .groupby('EDAD')['VALOR'].sum().reindex(edad_order, fill_value=0))
                emi_age_M = (emi.loc[(emi['ANO'] == flows_year) & (emi['SEXO'] == 1)]
                             .groupby('EDAD')['VALOR'].sum().reindex(edad_order, fill_value=0))
                net_M_annual = ratio_M * (imi_age_M - emi_age_M)
                net_M = PERIOD_YEARS * net_M_annual  # scale to period

                imi_age_F = (imi.loc[(imi['ANO'] == flows_year) & (imi['SEXO'] == 2)]
                             .groupby('EDAD')['VALOR'].sum().reindex(edad_order, fill_value=0))
                emi_age_F = (emi.loc[(emi['ANO'] == flows_year) & (emi['SEXO'] == 2)]
                             .groupby('EDAD')['VALOR'].sum().reindex(edad_order, fill_value=0))
                net_F_annual = ratio_F * (imi_age_F - emi_age_F)
                net_F = PERIOD_YEARS * net_F_annual  # scale to period

                # add half-period migration to exposures before rates; never let exposures fall <= 0
                conteos_all_M_p_t = (conteos_all_M_p_t + (net_M / 2.0)).clip(lower=1e-9)
                conteos_all_F_p_t = (conteos_all_F_p_t + (net_F / 2.0)).clip(lower=1e-9)

                # --- Life tables (5-year block)
                if ((death_choice == 'EEVV') and (year < 2024)) or ((death_choice != 'EEVV') and (year == 2018)):
                    lt_M_t = make_lifetable(fill_missing_age_bins(conteos_all_M_d_t).index,
                                            fill_missing_age_bins(conteos_all_M_p_t),
                                            fill_missing_age_bins(conteos_all_M_d_t))
                    lt_F_t = make_lifetable(fill_missing_age_bins(conteos_all_F_d_t).index,
                                            fill_missing_age_bins(conteos_all_F_p_t),
                                            fill_missing_age_bins(conteos_all_F_d_t))
                    lt_T_t = make_lifetable(fill_missing_age_bins(conteos_all_M_d_t).index,
                                            fill_missing_age_bins(conteos_all_M_p_t) + fill_missing_age_bins(conteos_all_F_p_t),
                                            fill_missing_age_bins(conteos_all_M_d_t) + fill_missing_age_bins(conteos_all_F_d_t))
                    if distribution is None:
                        lt_path = os.path.join('..', 'results', 'lifetables', DPTO, sample_type, death_choice, str(year))
                    else:
                        lt_path = os.path.join('..', 'results', 'lifetables', DPTO, sample_type, death_choice, distribution, str(year))
                    os.makedirs(lt_path, exist_ok=True)
                    lt_M_t.to_csv(os.path.join(lt_path, f'lt_M_t{suffix}.csv'))
                    lt_F_t.to_csv(os.path.join(lt_path, f'lt_F_t{suffix}.csv'))
                    lt_T_t.to_csv(os.path.join(lt_path, f'lt_T_t{suffix}.csv'))

                # --- ASFR (robust; fallbacks if TFR0 invalid)
                cutoff = last_obs_year_by_death[death_choice]
                key = (DPTO, death_choice)

                asfr_df = compute_asfr(
                    conteos_all_F_n_t.index,
                    pd.Series(conteos_all_F_p_t[conteos_all_F_p_t.index.isin(conteos_all_F_n_t.index)]),
                    pd.Series(conteos_all_F_n_t) + pd.Series(conteos_all_M_n_t)
                )
                asfr_df = asfr_df[['population', 'births', 'asfr']].astype(float)

                if year <= cutoff:
                    TFR0 = _tfr_from_asfr_df(asfr_df)
                    if not np.isfinite(TFR0) or TFR0 <= 0.0:
                        # fallback: use last known weights for this DPTO or national, scaled to baseline TFR
                        if key in asfr_weights and key in asfr_baseline:
                            w_norm = _normalize_weights_to(asfr_df.index, asfr_weights[key])
                            TFR0 = float(asfr_baseline[key]["TFR0"])
                            asfr_df["asfr"] = (w_norm * TFR0).astype(float)
                        else:
                            nat_key = ('total_nacional', death_choice)
                            if nat_key in asfr_weights and nat_key in asfr_baseline:
                                w_norm = _normalize_weights_to(asfr_df.index, asfr_weights[nat_key])
                                TFR0 = float(asfr_baseline[nat_key]["TFR0"])
                                asfr_df["asfr"] = (w_norm * TFR0).astype(float)
                            else:
                                raise ValueError(f"No usable ASFR for {key} in year {year} and no prior weights available.")
                        TFR0 = _tfr_from_asfr_df(asfr_df)

                    # store weights/baseline from observed (or rescued) year
                    w = asfr_df['asfr'] / TFR0
                    asfr_weights[key] = w
                    asfr_baseline[key] = {"year": year, "TFR0": TFR0}
                    asfr = asfr_df
                else:
                    if key not in asfr_weights or key not in asfr_baseline:
                        raise KeyError(f"No baseline ASFR weights stored for {key}; did you process year {cutoff} first?")
                    w = asfr_weights[key]
                    base = asfr_baseline[key]
                    step = year - base["year"]
                    TFR_t = _smooth_tfr(base["TFR0"], TFR_TARGET, CONV_YEARS, step, kind="exp", converge_frac=0.99)

                    proj_df = asfr_df.copy()
                    proj_df['population'] = np.nan
                    proj_df['births'] = np.nan
                    w_norm = _normalize_weights_to(proj_df.index, w)
                    proj_df['asfr'] = (w_norm * TFR_t).astype(float)

                    widths = _widths_from_index(proj_df.index)
                    chk = float(np.sum(proj_df['asfr'].values * widths))
                    if not np.isfinite(chk) or abs(chk - TFR_t) > 1e-6:
                        raise AssertionError(f"Normalization failed for {key} year {year}: {chk} vs {TFR_t}")
                    asfr_df = proj_df
                    asfr = proj_df

                if distribution is None:
                    asfr_path = os.path.join('..', 'results', 'asfr', DPTO, sample_type, str(year))
                else:
                    asfr_path = os.path.join('..', 'results', 'asfr', DPTO, sample_type, distribution, str(year))
                os.makedirs(asfr_path, exist_ok=True)
                asfr_df.to_csv(os.path.join(asfr_path, f'asfr{suffix}.csv'))

                # --- Projections (Leslie step = 5 years; pass full 5-yr net migration)
                if year == 2018:
                    L_MM, L_MF, L_FF, age_structures_df_M, age_structures_df_F, age_structures_df_T = make_projections(
                        net_F, net_M,
                        len(lt_F_t) - 1, 1, 2,
                        pd.Series(conteos_all_M_n_t),
                        pd.Series(conteos_all_F_n_t),
                        lt_F_t,
                        lt_M_t,
                        pd.Series(conteos_all_F_p_t),
                        pd.Series(conteos_all_M_p_t),
                        asfr,
                        100000,
                        year,
                        DPTO,
                        death_choice=death_choice
                    )
                else:
                    L_MM, L_MF, L_FF, age_structures_df_M, age_structures_df_F, age_structures_df_T = make_projections(
                        net_F, net_M,
                        len(lt_F_t) - 1, 1, 2,
                        pd.Series(conteos_all_M_n_t),
                        pd.Series(conteos_all_F_n_t),
                        lt_F_t,
                        lt_M_t,
                        pd.Series(conteos_all_F_p_t_updated),
                        pd.Series(conteos_all_M_p_t_updated),
                        asfr,
                        100000,
                        year,
                        DPTO,
                        death_choice=death_choice
                    )
                save_LL(L_MM, L_MF, L_FF, death_choice, DPTO, sample_type, distribution, suffix, year)

                proj_F = pd.concat([proj_F, age_structures_df_F], axis=0, ignore_index=True, sort=False)
                proj_M = pd.concat([proj_M, age_structures_df_M], axis=0, ignore_index=True, sort=False)
                proj_T = pd.concat([proj_T, age_structures_df_T], axis=0, ignore_index=True, sort=False)

        save_projections(proj_F, proj_M, proj_T, sample_type, distribution, suffix, death_choice, year)


if __name__ == '__main__':
    # Configuration
    if len(sys.argv) > 1 and sys.argv[1] == "draws":
        print("We'll be running this with draws")
        num_draws = 1000
        dist_types = ['uniform', 'pert', 'beta', 'normal']
        tasks = []
        for dist in dist_types:
            for i in range(num_draws):
                label = f'{dist}_draw_{i}'
                tasks.append(('draw', dist, label))
    else:
        print("We'll be running this without draws")
        tasks = [
            ('mid', None, 'mid_omissions'),
            ('low', None, 'low_omissions'),
            ('high', None, 'high_omissions'),
        ]

    # 5-year steps (2018, 2023, 2028, ..., 2070)
    projection_range = range(2018, 2071, 5)

    # Load data
    data = load_all_data()
    conteos = data['conteos']
    # (these are your migration tables)
    emi = conteos[conteos['VARIABLE'] == 'crt_visa_F_emigracion']
    imi = conteos[conteos['VARIABLE'] == 'crt_visa_F_inmigracion']

    def _execute_task(args):
        sample_type, dist, label = args
        seed = zlib.adler32(label.encode('utf8')) & 0xFFFFFFFF
        np.random.seed(seed)

        df = conteos.copy()
        df['VALOR_withmissing'] = df['VALOR']
        df['VALOR_corrected'] = np.nan

        processed_subsets = []
        for var in ['defunciones', 'nacimientos', 'poblacion_total']:
            mask = df['VARIABLE'] == var
            df_var = df.loc[mask].copy()
            df_var = allocate_and_drop_missing_age(df_var)
            df_var.loc[:, 'VALOR_corrected'] = correct_valor_for_omission(
                df_var,
                sample_type,
                distribution=dist,
                valor_col='VALOR_withmissing'
            )
            processed_subsets.append(df_var)

        df = pd.concat(processed_subsets, axis=0, ignore_index=True)
        df = df[df['EDAD'].notna()].copy()

        if dist is None:
            main_wrapper(df, emi, imi, projection_range, label)
        else:
            main_wrapper(df, emi, imi, projection_range, 'draw', dist, label)
        return label

    # Parallel execution
    with Pool(1) as pool:
        for _ in tqdm(pool.imap_unordered(_execute_task, tasks), total=len(tasks), desc='Tasks'):
            pass
