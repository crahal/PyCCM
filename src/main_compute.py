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
from projections import make_projections, save_LL
from data_loaders import load_all_data, correct_valor_for_omission, allocate_and_drop_missing_age

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
    proj_F = pd.DataFrame()
    proj_M = pd.DataFrame()
    proj_T = pd.DataFrame()
    for year in projection_range:
        for DPTO in DTPO_list:
            print(year, DPTO)
            if DPTO != 'total_nacional':
                conteos_all = conteos[conteos['DPTO_NOMBRE'] == DPTO]
            else:
                conteos_all = conteos[conteos['DPTO_NOMBRE'] != DPTO]
            if year>2023:
                conteos_all = conteos_all[conteos_all['ANO'] == 2023]
            else:
                conteos_all = conteos_all[conteos_all['ANO'] == year]
            conteos_all_M = conteos_all[conteos_all['SEXO'] == 1.0]
            conteos_all_F = conteos_all[conteos_all['SEXO'] == 2.0]

            conteos_all_M_n = conteos_all_M[(conteos_all_M['VARIABLE'] == 'nacimientos') &
                                            (conteos_all_M['FUENTE'] == 'EEVV')]
            conteos_all_F_n = conteos_all_F[(conteos_all_F['VARIABLE'] == 'nacimientos') &
                                            (conteos_all_F['FUENTE'] == 'EEVV')]

            conteos_all_M_d = conteos_all_M[(conteos_all_M['VARIABLE'] == 'defunciones') &
                                            (conteos_all_M['FUENTE'] == 'EEVV')]
            conteos_all_F_d = conteos_all_F[(conteos_all_F['VARIABLE'] == 'defunciones') &
                                            (conteos_all_F['FUENTE'] == 'EEVV')]

            if year == 2018:
                conteos_all_M_p = conteos_all_M[(conteos_all_M['VARIABLE'] == 'poblacion_total') &
                                                (conteos_all_M['FUENTE'] == 'censo_2018')]
                conteos_all_F_p = conteos_all_F[(conteos_all_F['VARIABLE'] == 'poblacion_total') &
                                                (conteos_all_F['FUENTE'] == 'censo_2018')]
            else:
                # USE PROJECTIONS FOR POPULATION
                conteos_all_F_p = proj_F[(proj_F['year'] == year) & (proj_F['DPTO_NOMBRE'] == DPTO)]
                conteos_all_M_p = proj_M[(proj_M['year'] == year) & (proj_M['DPTO_NOMBRE'] == DPTO)]
            if DPTO == 'total_nacional':
                # Aggregate across all departments
                conteos_all_M_n_t = conteos_all_M_n.groupby(['EDAD'])['VALOR_corrected'].sum()
                conteos_all_F_n_t = conteos_all_F_n.groupby(['EDAD'])['VALOR_corrected'].sum()
                conteos_all_M_d_t = conteos_all_M_d.groupby(['EDAD'])['VALOR_corrected'].sum()
                conteos_all_F_d_t = conteos_all_F_d.groupby(['EDAD'])['VALOR_corrected'].sum()
                conteos_all_M_p_t = conteos_all_M_p.groupby(['EDAD'])['VALOR_corrected'].sum()
                conteos_all_F_p_t = conteos_all_F_p.groupby(['EDAD'])['VALOR_corrected'].sum()
            else:
                # Use existing index-aligned series for each department
                conteos_all_M_n_t = conteos_all_M_n.set_index('EDAD')['VALOR_corrected']
                conteos_all_F_n_t = conteos_all_F_n.set_index('EDAD')['VALOR_corrected']
                conteos_all_M_d_t = conteos_all_M_d.set_index('EDAD')['VALOR_corrected']
                conteos_all_F_d_t = conteos_all_F_d.set_index('EDAD')['VALOR_corrected']
                conteos_all_M_p_t = conteos_all_M_p.set_index('EDAD')['VALOR_corrected']
                conteos_all_F_p_t = conteos_all_F_p.set_index('EDAD')['VALOR_corrected']
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
                lt_path = os.path.join('..', 'results', 'lifetables', DPTO, sample_type)
            else:
                lt_path = os.path.join('..', 'results', 'lifetables', DPTO, sample_type, distribution)
            os.makedirs(lt_path, exist_ok=True)
            suffix = f"_{draw}" if distribution is not None else ''
            lt_M_t.to_csv(os.path.join(lt_path, f'lt_M_t{suffix}.csv'))
            lt_F_t.to_csv(os.path.join(lt_path, f'lt_F_t{suffix}.csv'))
            lt_T_t.to_csv(os.path.join(lt_path, f'lt_T_t{suffix}.csv'))

            # Compute ASFR
            asfr = compute_asfr(conteos_all_F_n_t.index,
                                pd.Series(conteos_all_F_p_t[
                                conteos_all_F_p_t.index.isin(conteos_all_F_n_t.index)]),
                                pd.Series(conteos_all_F_n_t) + pd.Series(conteos_all_M_n_t))

            # Save ASFR
            if distribution is None:
                asfr_path = os.path.join('..', 'results', 'asfr', DPTO, sample_type)
            else:
                asfr_path = os.path.join('..', 'results', 'asfr', DPTO, sample_type, distribution)
            os.makedirs(asfr_path, exist_ok=True)
            asfr.to_csv(os.path.join(asfr_path, f'asfr{suffix}.csv'))

            # Projections
            _, _, _, age_structures_df_M, age_structures_df_F, age_structures_df_T = make_projections(
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
                DPTO
            )

            # Save age structures
            proj_F = pd.concat([proj_F, age_structures_df_F], axis=0, ignore_index=True, sort=False)
            proj_M = pd.concat([proj_M, age_structures_df_M], axis=0, ignore_index=True, sort=False)
            proj_T = pd.concat([proj_T, age_structures_df_T], axis=0, ignore_index=True, sort=False)
    proj_F.to_csv('proj_f_temp.csv')

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
    projection_range = range(2018, 2026)
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
#    with Pool(processes=cpu_count()-1) as pool:
    with Pool(1) as pool:
        for _ in tqdm(pool.imap_unordered(_execute_task, tasks), total=len(tasks), desc='Tasks'):
            pass
