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
from projections import make_projections, save_LL, save_projections
from data_loaders import load_all_data, correct_valor_for_omission, allocate_and_drop_missing_age


# --- smooth TFR paths ---------------------------------------------------------

def _exp_tfr(TFR0: float, target: float, years: int, step: int,
             converge_frac: float = 0.99) -> float:
    """
    Monotone exponential approach to 'target':
        TFR(t) = target + (TFR0 - target) * exp(-kappa * t),  t = step (years).
    'converge_frac' sets how close (in fraction of the initial gap) we are
    to the target at 'years' (e.g., 0.99 → 99% of the gap removed).
    """
    t = max(step, 0)
    gap0 = float(TFR0 - target)
    if gap0 == 0.0:
        return float(target)
    # choose kappa so that |gap(years)| = (1 - converge_frac) * |gap0|
    # i.e. exp(-kappa*years) = 1 - converge_frac
    eps = 1e-12
    q = max(1.0 - converge_frac, eps)              # residual fraction at 'years'
    kappa = -np.log(q) / max(years, 1)             # rate parameter
    return float(target + gap0 * np.exp(-kappa * t))


def _logistic_tfr(TFR0: float, target: float, years: int, step: int,
                  mid_frac: float = 0.5, steepness: float | None = None) -> float:
    """
    Logistic (sigmoid) path between TFR0 (at t=0) and 'target' (asymptote).
    Center (inflection) is at t0 = mid_frac * years; if 'steepness' is None,
    it is chosen so that the value at t=years is within ~1% of the target.
        TFR(t) = target + (TFR0 - target) / (1 + exp[s*(t - t0)])
                 * (1 + exp[-s*t0])                      (re-scaled so TFR(0)=TFR0)
    """
    t = max(step, 0.0)
    t0 = float(mid_frac) * max(years, 1)
    # pick s so that distance at the horizon is about 1% of initial gap
    if steepness is None:
        r = 100.0  # ~1% residual at the horizon
        steepness = (np.log(r) - np.log(1 + np.exp(-t0))) / max(years - t0, 1e-9)

    s = float(steepness)
    gap0 = float(TFR0 - target)
    # scale factor so that TFR(0) = TFR0 exactly
    scale = 1.0 + np.exp(-s * t0)
    return float(target + gap0 * (scale / (1.0 + np.exp(s * (t - t0)))))


def _smooth_tfr(TFR0: float, target: float, years: int, step: int,
                kind: str = "exp", **kwargs) -> float:
    """
    Dispatcher: 'exp' (default) or 'logistic'.
    """
    if kind == "exp":
        return _exp_tfr(TFR0, target, years, step, **kwargs)
    elif kind == "logistic":
        return _logistic_tfr(TFR0, target, years, step, **kwargs)
    else:
        raise ValueError(f"Unknown TFR path kind: {kind!r}")


def _bin_width(label: str) -> float:
    """Width Δa of an age bin label like '15-19', '45-49', '80+'."""
    s = str(label)
    if '-' in s:
        lo, hi = s.split('-')
        return float(hi) - float(lo) + 1.0  # 14-10+1 = 5
    if s.endswith('+'):
        return 5.0
    return 1.0

def _widths_from_index(idx) -> np.ndarray:
    return np.array([_bin_width(x) for x in idx], dtype=float)

def _tfr_from_asfr_df(asfr_df: pd.DataFrame) -> float:
    """Integrate the *asfr* column with bin widths Δa to get TFR."""
    s = asfr_df['asfr'].astype(float)
    widths = _widths_from_index(s.index)
    return float(np.sum(s.values * widths))

def _linear_tfr(TFR0: float, target: float, years: int, step: int) -> float:
    """Linear path hitting target at exactly `years`."""
    u = min(max(step / years, 0.0), 1.0)
    return TFR0 + (target - TFR0) * u



def fill_missing_age_bins(s: pd.Series) -> pd.Series:
    """
    Ensures that all expected age bins are present in the given Series.
    Missing bins are filled with 0.

    Parameters
    ----------
    s : pd.Series
        A pandas Series indexed by age bins (e.g. '0-4', '5-9', ..., '80+').

    Returns
    -------
    pd.Series
        A Series with all expected bins present, missing values filled with 0.
    """
    expected_bins = [
        "0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
        "40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"
    ]

    return s.reindex(expected_bins, fill_value=0)



def main_wrapper(conteos, projection_range, sample_type, distribution=None, draw=None):
    DTPO_list = list(conteos['DPTO_NOMBRE'].unique()) + ['total_nacional']
    suffix = f"_{draw}" if distribution is not None else ''

    asfr_weights = {}      # key: (DPTO, death_choice) -> pd.Series w(a) with sum(w*Δa)=1
    asfr_baseline = {}     # key -> dict(year=y0, TFR0=TFR0)

    # choose the last observed year for births by source:
    last_obs_year_by_death = {'EEVV': 2023, 'censo_2018': 2018, 'midpoint': 2018}

    TFR_TARGET = 1.5
    CONV_YEARS = 50
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
                    # use EEVV for deaths
                   if year>2023:
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
                    # use midpoint between EEVV and censo_2018 for deaths
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
                    elif (year>2018 and year<=2023):
                        # USE PROJECTIONS FOR POPULATION
                        conteos_all_F_p = proj_F[(proj_F['year'] == year) &
                            (proj_F['DPTO_NOMBRE'] == DPTO) &
                            (proj_M['death_choice'] == death_choice)]
                        conteos_all_M_p = proj_M[(proj_M['year'] == year) &
                            (proj_M['DPTO_NOMBRE'] == DPTO) &
                            (proj_M['death_choice'] == death_choice)]


                if year>2018:
                    conteos_all_F_p_updated = proj_F[(proj_F['year'] == year) &
                        (proj_F['DPTO_NOMBRE'] == DPTO) &
                        (proj_M['death_choice'] == death_choice)]
                    conteos_all_M_p_updated = proj_M[(proj_M['year'] == year) &
                        (proj_M['DPTO_NOMBRE'] == DPTO) &
                        (proj_M['death_choice'] == death_choice)]

                if DPTO == 'total_nacional':
                    # Aggregate across all departments
                    conteos_all_M_n_t = conteos_all_M_n.groupby(['EDAD'])['VALOR_corrected'].sum()
                    conteos_all_F_n_t = conteos_all_F_n.groupby(['EDAD'])['VALOR_corrected'].sum()
                    conteos_all_M_d_t = conteos_all_M_d.groupby(['EDAD'])['VALOR_corrected'].sum()
                    conteos_all_F_d_t = conteos_all_F_d.groupby(['EDAD'])['VALOR_corrected'].sum()
                    conteos_all_M_p_t = conteos_all_M_p.groupby(['EDAD'])['VALOR_corrected'].sum()
                    conteos_all_F_p_t = conteos_all_F_p.groupby(['EDAD'])['VALOR_corrected'].sum()
                    if year>2018:
                        conteos_all_F_p_t_updated = conteos_all_F_p_updated.groupby(['EDAD'])['VALOR_corrected'].sum()
                        conteos_all_M_p_t_updated = conteos_all_M_p_updated.groupby(['EDAD'])['VALOR_corrected'].sum()
                else:
                    # Use existing index-aligned series for each department
                    conteos_all_M_n_t = conteos_all_M_n.set_index('EDAD')['VALOR_corrected']
                    conteos_all_F_n_t = conteos_all_F_n.set_index('EDAD')['VALOR_corrected']
                    conteos_all_M_d_t = conteos_all_M_d.set_index('EDAD')['VALOR_corrected']
                    conteos_all_F_d_t = conteos_all_F_d.set_index('EDAD')['VALOR_corrected']
                    conteos_all_M_p_t = conteos_all_M_p.set_index('EDAD')['VALOR_corrected']
                    conteos_all_F_p_t = conteos_all_F_p.set_index('EDAD')['VALOR_corrected']
                    if year>2018:
                        conteos_all_M_p_t_updated = conteos_all_M_p_updated.set_index('EDAD')['VALOR_corrected']
                        conteos_all_F_p_t_updated = conteos_all_F_p_updated.set_index('EDAD')['VALOR_corrected']

                lt_M_t = make_lifetable(fill_missing_age_bins(conteos_all_M_d_t).index,
                                        fill_missing_age_bins(conteos_all_M_p_t),
                                        fill_missing_age_bins(conteos_all_M_d_t)
                )
                lt_F_t = make_lifetable(fill_missing_age_bins(conteos_all_F_d_t).index,
                                        fill_missing_age_bins(conteos_all_F_p_t),
                                        fill_missing_age_bins(conteos_all_F_d_t)
                )
                lt_T_t = make_lifetable(fill_missing_age_bins(conteos_all_M_d_t).index,
                                        fill_missing_age_bins(conteos_all_M_p_t) + fill_missing_age_bins(conteos_all_F_p_t),
                                        fill_missing_age_bins(conteos_all_M_d_t) + fill_missing_age_bins(conteos_all_F_d_t))
                # Save lifetables
                if distribution is None:
                    lt_path = os.path.join('..', 'results', 'lifetables', DPTO, sample_type, death_choice, str(year))
                else:
                    lt_path = os.path.join('..', 'results', 'lifetables', DPTO, sample_type, death_choice, distribution, str(year))
                os.makedirs(lt_path, exist_ok=True)
                lt_M_t.to_csv(os.path.join(lt_path, f'lt_M_t{suffix}.csv'))
                lt_F_t.to_csv(os.path.join(lt_path, f'lt_F_t{suffix}.csv'))
                lt_T_t.to_csv(os.path.join(lt_path, f'lt_T_t{suffix}.csv'))

                # Compute ASFR
                cutoff = last_obs_year_by_death[death_choice]
                key = (DPTO, death_choice)

                # compute the observed DF (population, births, asfr)
                asfr_df = compute_asfr(
                    conteos_all_F_n_t.index,
                    pd.Series(conteos_all_F_p_t[conteos_all_F_p_t.index.isin(conteos_all_F_n_t.index)]),
                    pd.Series(conteos_all_F_n_t) + pd.Series(conteos_all_M_n_t)
                )
                # ensure dtype and ordering
                asfr_df = asfr_df[['population', 'births', 'asfr']].astype(float)

                if year <= cutoff:
                    # OBSERVED: save as is; also (re)write frozen weights at the final observed year
                    TFR0 = _tfr_from_asfr_df(asfr_df)
                    if TFR0 <= 0 or not np.isfinite(TFR0):
                        raise ValueError(f"TFR0 not positive/finite for {key} in year {year}")
                    w = asfr_df['asfr'] / TFR0                # w(a), Σ w(a) Δa = 1
                    asfr_weights[key] = w
                    asfr_baseline[key] = {"year": year, "TFR0": TFR0}
                    asfr = asfr_df                     # (series) for make_projections below
                else:
                    # PROJECTED: scale frozen weights by TFR_t, keep same DataFrame structure
                    if key not in asfr_weights:
                        raise KeyError(f"No baseline ASFR weights stored for {key}; "
                                       f"did you process year {cutoff} first?")
                    w = asfr_weights[key]
                    base = asfr_baseline[key]
                    step = year - base["year"]
                    TFR_t = _smooth_tfr(base["TFR0"], TFR_TARGET, CONV_YEARS, step, kind="exp", converge_frac=0.99)


                    proj_df = asfr_df.copy()
                    proj_df['population'] = np.nan           # keep columns, avoid invented counts
                    proj_df['births'] = np.nan
                    proj_df['asfr'] = (w * TFR_t).astype(float)

                    # rigorous normalization check
                    widths = _widths_from_index(proj_df.index)
                    chk = float(np.sum(proj_df['asfr'].values * widths))
                    if abs(chk - TFR_t) > 1e-6:
                        raise AssertionError(f"Normalization failed for {key} year {year}: {chk} vs {TFR_t}")

                    # save for I/O and pass just the Series to the projector
                    asfr_df = proj_df
                    asfr = proj_df
                # Save ASFR (unchanged structure on disk)
                if distribution is None:
                    asfr_path = os.path.join('..', 'results', 'asfr', DPTO, sample_type, str(year))
                else:
                    asfr_path = os.path.join('..', 'results', 'asfr', DPTO, sample_type, distribution, str(year))
                os.makedirs(asfr_path, exist_ok=True)
                asfr_df.to_csv(os.path.join(asfr_path, f'asfr{suffix}.csv'))

                # Projections
                #
                if year == 2018:
                     L_MM, L_MF, L_FF, age_structures_df_M, age_structures_df_F, age_structures_df_T = make_projections(
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
                # Save age structures
                proj_F = pd.concat([proj_F, age_structures_df_F], axis=0, ignore_index=True, sort=False)
                proj_M = pd.concat([proj_M, age_structures_df_M], axis=0, ignore_index=True, sort=False)
                proj_T = pd.concat([proj_T, age_structures_df_T], axis=0, ignore_index=True, sort=False)

        save_projections(proj_F, proj_M, proj_T, sample_type, distribution, suffix, death_choice, year)


if __name__ == '__main__':
    # Configuration

    if sys.argv[1] == "draws":
        print("We'll be running this with draws")
        num_draws = 1000
        dist_types = ['uniform', 'pert', 'beta', 'normal']
        for dist in dist_types:
            for i in range(num_draws):
                label = f'{dist}_draw_{i}'
                tasks.append(('draw', dist, label))
    else:
        print("We'll be running this without draws")
    projection_range = range(2018, 2070)
    # Load data
    data = load_all_data()
    conteos = data['conteos']
    # Prepare tasks
    tasks = [
        ('mid', None, 'mid_omissions'),
        ('low', None, 'low_omissions'),
        ('high', None, 'high_omissions'),
    ]

    def _execute_task(args):
        sample_type, dist, label = args
        # Turn the task’s label into a seed
        seed = zlib.adler32(label.encode('utf8')) & 0xFFFFFFFF
        np.random.seed(seed)
        df = conteos.copy()
        # 0) Initialize
        df['VALOR_withmissing'] = df['VALOR']
        df['VALOR_corrected'] = np.nan
        # 1) Process each variable separately
        processed_subsets = []
        for var in ['defunciones', 'nacimientos', 'poblacion_total']:
            # a) select only rows of this variable
            mask = df['VARIABLE'] == var
            df_var = df.loc[mask].copy()

            # b) allocate missing ages in-place on this subset
            df_var = allocate_and_drop_missing_age(df_var)

            # c) correct for omissions on this subset
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
            main_wrapper(df, projection_range, label)
        else:
            main_wrapper(df, projection_range, 'draw', dist, label)
        return label

    # Parallel execution
    with Pool(processes=cpu_count()-5) as pool:
#    with Pool(1) as pool:
        for _ in tqdm(pool.imap_unordered(_execute_task, tasks), total=len(tasks), desc='Tasks'):
            pass
