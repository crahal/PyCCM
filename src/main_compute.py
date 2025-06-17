import pyreadr
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from mortality import make_lifetable
from fertility import compute_asfr
from projections import make_projections
import matplotlib.pyplot as plt
from data_loaders import load_all_data, correct_valor_for_omission


def main_wrapper(conteos, sample_type, distribution=None, draw=None):
    for DPTO in conteos['DPTO_NOMBRE'].unique():
        if DPTO != 'total_nacional':
            conteos_all = conteos[conteos['DPTO_NOMBRE'] == DPTO]
        else:
            conteos_all = conteos[conteos['DPTO_NOMBRE'] != DPTO]
        conteos_all_2018 = conteos_all[conteos_all['ANO'] == 2018]
        conteos_all_2018_M = conteos_all_2018[conteos_all_2018['SEXO'] == 1.0]
        conteos_all_2018_F = conteos_all_2018[conteos_all_2018['SEXO'] == 2.0]

        conteos_all_2018_t_n = conteos_all_2018[(conteos_all_2018['VARIABLE'] == 'nacimientos') &
                                                (conteos_all_2018['FUENTE'] == 'EEVV')]
        conteos_all_2018_t_d = conteos_all_2018[(conteos_all_2018['VARIABLE'] == 'defunciones') &
                                                (conteos_all_2018['FUENTE'] == 'EEVV')]
        conteos_all_2018_t_d_census = conteos_all_2018[(conteos_all_2018['VARIABLE'] == 'defunciones') &
                                                       (conteos_all_2018['FUENTE'] == 'censo_2018')]
        conteos_all_2018_t_p = conteos_all_2018[(conteos_all_2018['VARIABLE'] == 'poblacion_total') &
                                                (conteos_all_2018['FUENTE'] == 'censo_2018')]

        conteos_all_2018_M_n = conteos_all_2018_M[(conteos_all_2018_M['VARIABLE'] == 'nacimientos') &
                                                  (conteos_all_2018_M['FUENTE'] == 'EEVV')]
        conteos_all_2018_M_d = conteos_all_2018_M[(conteos_all_2018_M['VARIABLE'] == 'defunciones') &
                                                  (conteos_all_2018_M['FUENTE'] == 'EEVV')]
        conteos_all_2018_M_d_census = conteos_all_2018_M[(conteos_all_2018_M['VARIABLE'] == 'defunciones') &
                                                         (conteos_all_2018_M['FUENTE'] == 'censo_2018')]
        conteos_all_2018_M_p = conteos_all_2018_M[(conteos_all_2018_M['VARIABLE'] == 'poblacion_total') &
                                                  (conteos_all_2018_M['FUENTE'] == 'censo_2018')]

        conteos_all_2018_F_n = conteos_all_2018_F[(conteos_all_2018_F['VARIABLE'] == 'nacimientos') &
                                                  (conteos_all_2018_F['FUENTE'] == 'EEVV')]
        conteos_all_2018_F_d = conteos_all_2018_F[(conteos_all_2018_F['VARIABLE'] == 'defunciones') &
                                                  (conteos_all_2018_F['FUENTE'] == 'EEVV')]
        conteos_all_2018_F_d_census = conteos_all_2018_F[(conteos_all_2018_F['VARIABLE'] == 'defunciones') &
                                                         (conteos_all_2018_F['FUENTE'] == 'censo_2018')]
        conteos_all_2018_F_p = conteos_all_2018_F[(conteos_all_2018_F['VARIABLE'] == 'poblacion_total') &
                                                  (conteos_all_2018_F['FUENTE'] == 'censo_2018')]
        if DPTO == 'total_nacional':
            conteos_all_2018_M_n_t = conteos_all_2018_M_n.groupby(['EDAD'])['VALOR_corrected'].sum()
            conteos_all_2018_M_d_t = conteos_all_2018_M_d.groupby(['EDAD'])['VALOR_corrected'].sum()
            conteos_all_2018_M_d_t_census = conteos_all_2018_M_d_census.groupby(['EDAD'])['VALOR_corrected'].sum()
            conteos_all_2018_M_p_t = conteos_all_2018_M_p.groupby(['EDAD'])['VALOR_corrected'].sum()

            conteos_all_2018_F_n_t = conteos_all_2018_F_n.groupby(['EDAD'])['VALOR_corrected'].sum()
            conteos_all_2018_F_d_t = conteos_all_2018_F_d.groupby(['EDAD'])['VALOR_corrected'].sum()
            conteos_all_2018_F_d_t_census = conteos_all_2018_F_d_census.groupby(['EDAD'])['VALOR_corrected'].sum()
            conteos_all_2018_F_p_t = conteos_all_2018_F_p.groupby(['EDAD'])['VALOR_corrected'].sum()

            conteos_all_2018_T_n_t = conteos_all_2018_t_n.groupby(['EDAD'])['VALOR_corrected'].sum()
            conteos_all_2018_T_d_t = conteos_all_2018_t_d.groupby(['EDAD'])['VALOR_corrected'].sum()
            conteos_all_2018_T_d_t_census = conteos_all_2018_t_d_census.groupby(['EDAD'])['VALOR_corrected'].sum()
            conteos_all_2018_T_p_t = conteos_all_2018_t_p.groupby(['EDAD'])['VALOR_corrected'].sum()
        else:
            conteos_all_2018_M_n_t = conteos_all_2018_M_n.set_index('EDAD')['VALOR_corrected']
            conteos_all_2018_M_d_t = conteos_all_2018_M_d.set_index('EDAD')['VALOR_corrected']
            conteos_all_2018_M_d_t_census = conteos_all_2018_M_d_census.set_index('EDAD')['VALOR_corrected']
            conteos_all_2018_M_p_t = conteos_all_2018_M_p.set_index('EDAD')['VALOR_corrected']

            conteos_all_2018_F_n_t = conteos_all_2018_F_n.set_index('EDAD')['VALOR_corrected']
            conteos_all_2018_F_d_t = conteos_all_2018_F_d.set_index('EDAD')['VALOR_corrected']
            conteos_all_2018_F_d_t_census = conteos_all_2018_F_d_census.set_index('EDAD')['VALOR_corrected']
            conteos_all_2018_F_p_t = conteos_all_2018_F_p.set_index('EDAD')['VALOR_corrected']

            conteos_all_2018_T_n_t = conteos_all_2018_t_n.set_index('EDAD')['VALOR_corrected']
            conteos_all_2018_T_d_t = conteos_all_2018_t_d.set_index('EDAD')['VALOR_corrected']
            conteos_all_2018_T_d_t_census = conteos_all_2018_t_d_census.set_index('EDAD')['VALOR_corrected']
            conteos_all_2018_T_p_t = conteos_all_2018_t_p.set_index('EDAD')['VALOR_corrected']

        # align the DFs for defunciones to plot later.
        keys = ['DPTO_NOMBRE', 'DPTO_CODIGO', 'ANO', 'SEXO', 'EDAD', 'VARIABLE']
        df_aligned_F = (
            conteos_all_2018_F_d_census
            .merge(
                conteos_all_2018_F_d,
                on=keys,
                how='outer',
                suffixes=('_censo', '_EEVV')
            )
            .fillna({'VALOR_corrected_censo': 0,
                     'VALOR_corrected_EEVV': 0})
        )

        df_aligned_M = (
            conteos_all_2018_M_d_census
            .merge(
                conteos_all_2018_M_d,
                on=keys,
                how='outer',
                suffixes=('_censo', '_EEVV')
            )
            .fillna({'VALOR_corrected_censo': 0,
                     'VALOR_corrected_EEVV': 0})
        )

        # Make lifetables, save them and then merge them
        lt_2018_M_t = make_lifetable(conteos_all_2018_M_d_t.index,
                                     conteos_all_2018_M_p_t,
                                     conteos_all_2018_M_d_t)
        lt_2018_F_t = make_lifetable(conteos_all_2018_F_d_t.index,
                                     conteos_all_2018_F_p_t,
                                     conteos_all_2018_F_d_t)
        lt_2018_T_t = make_lifetable(conteos_all_2018_T_d_t.index,
                                     conteos_all_2018_T_p_t,
                                     conteos_all_2018_T_d_t)
        if distribution is None:
            lt_path = os.path.join('..', 'results', 'lifetables', DPTO, sample_type)
            os.makedirs(lt_path, exist_ok=True)
            lt_2018_M_t.to_csv(os.path.join(lt_path, 'lt_2018_M_t.csv'))
            lt_2018_F_t.to_csv(os.path.join(lt_path, 'lt_2018_F_t.csv'))
            lt_2018_T_t.to_csv(os.path.join(lt_path, 'lt_2018_T_t.csv'))
        else:
            lt_path = os.path.join('..', 'results', 'lifetables', DPTO, sample_type, distribution)
            os.makedirs(lt_path, exist_ok=True)
            lt_2018_M_t.to_csv(os.path.join(lt_path, 'lt_2018_M_t' + '_' + str(draw) + '.csv'))
            lt_2018_F_t.to_csv(os.path.join(lt_path, 'lt_2018_F_t.csv'))
            lt_2018_T_t.to_csv(os.path.join(lt_path, 'lt_2018_T_t.csv'))

        '''
        lt_2018_M_t_census = make_lifetable(conteos_all_2018_M_d_t_census.index,
                                     conteos_all_2018_M_p_t,
                                     conteos_all_2018_M_d_t_census)
        lt_2018_F_t_census = make_lifetable(conteos_all_2018_F_d_t_census.index,
                                     conteos_all_2018_F_p_t,
                                     conteos_all_2018_F_d_t_census)
        lt_2018_T_t_census = make_lifetable(conteos_all_2018_T_d_t_census.index,
                                     conteos_all_2018_T_p_t,
                                     conteos_all_2018_T_d_t_census)
        lt_combined_M = (
            lt_2018_M_t_census.add_suffix('_censo')
                .join(lt_2018_M_t.add_suffix('_EEVV'),
                      how='outer')
        )

        lt_combined_F = (
            lt_2018_F_t_census.add_suffix('_censo')
                .join(lt_2018_F_t.add_suffix('_EEVV'),
                      how='outer')
        )
        '''
        # Calculate the ASFR
        asfr_2018 = compute_asfr(conteos_all_2018_F_n_t.index,
                                 conteos_all_2018_F_p_t[
                                     conteos_all_2018_F_p_t.index.isin(conteos_all_2018_F_n_t.index)],
                                 conteos_all_2018_F_n_t + conteos_all_2018_M_n_t)

        if distribution is None:
            asfr_path = os.path.join('..', 'results', 'asfr', DPTO, sample_type)
            os.makedirs(asfr_path, exist_ok=True)
            asfr_2018.to_csv(os.path.join(asfr_path, 'asfr_2018.csv'))
        else:
            asfr_path = os.path.join('..', 'results', 'asfr', DPTO, sample_type, distribution)
            os.makedirs(asfr_path, exist_ok=True)
            asfr_2018.to_csv(os.path.join(lt_path, 'asfr_2018' + '_' + str(draw) + '.csv'))

        L_MM, L_MF, L_FF, age_structures_df_M, age_structures_df_F = make_projections(len(lt_2018_F_t) - 1, 50, 2,
                                                                                      conteos_all_2018_M_n_t,
                                                                                      conteos_all_2018_F_n_t,
                                                                                      lt_2018_F_t,
                                                                                      lt_2018_M_t,
                                                                                      conteos_all_2018_F_p_t,
                                                                                      conteos_all_2018_M_p_t,
                                                                                      asfr_2018,
                                                                                      100000)
        if distribution is None:
            L_MM_path = os.path.join('..', 'results', 'projections', DPTO, 'L_MM', sample_type)
            L_MF_path = os.path.join('..', 'results', 'projections', DPTO, 'L_MF', sample_type)
            L_FF_path = os.path.join('..', 'results', 'projections', DPTO, 'L_FF', sample_type)
            age_structures_m_path = os.path.join('..', 'results', 'projections', DPTO, 'age_structures_df_M', sample_type)
            age_structures_f_path = os.path.join('..', 'results', 'projections', DPTO, 'age_structures_df_F', sample_type)
            os.makedirs(L_MM_path, exist_ok=True)
            os.makedirs(L_MF_path, exist_ok=True)
            os.makedirs(L_FF_path, exist_ok=True)
            os.makedirs(age_structures_m_path, exist_ok=True)
            os.makedirs(age_structures_f_path, exist_ok=True)
            pd.DataFrame(L_MM).to_csv(os.path.join(L_MM_path, 'LMM.csv'))
            pd.DataFrame(L_MF).to_csv(os.path.join(L_MF_path, 'LMF.csv'))
            pd.DataFrame(L_FF).to_csv(os.path.join(L_FF_path, 'LFF.csv'))
            age_structures_df_M.to_csv(os.path.join(age_structures_m_path, 'age_structures_df_M.csv'))
            age_structures_df_F.to_csv(os.path.join(age_structures_f_path, 'age_structures_df_F.csv'))
        else:
            L_MM_path = os.path.join('..', 'results', 'projections', DPTO, 'L_MM', sample_type, distribution)
            L_MF_path = os.path.join('..', 'results', 'projections', DPTO, 'L_MF', sample_type, distribution)
            L_FF_path = os.path.join('..', 'results', 'projections', DPTO, 'L_FF', sample_type, distribution)
            age_structures_m_path = os.path.join('..', 'results', 'projections', DPTO, 'age_structures_df_M', sample_type, distribution)
            age_structures_f_path = os.path.join('..', 'results', 'projections', DPTO, 'age_structures_df_F', sample_type, distribution)
            os.makedirs(L_MM_path, exist_ok=True)
            os.makedirs(L_MF_path, exist_ok=True)
            os.makedirs(L_FF_path, exist_ok=True)
            os.makedirs(age_structures_m_path, exist_ok=True)
            os.makedirs(age_structures_f_path, exist_ok=True)
            pd.DataFrame(L_MM).to_csv(os.path.join(L_MM_path, 'LMM' + ' ' + str(draw) + '.csv'))
            pd.DataFrame(L_FF).to_csv(os.path.join(L_MF_path, 'LMF' + ' ' + str(draw) + '.csv'))
            pd.DataFrame(L_FF).to_csv(os.path.join(L_FF_path, 'LFF' + ' ' + str(draw) + '.csv'))
            age_structures_df_M.to_csv(
                os.path.join(age_structures_m_path, 'age_structures_df_M.csv' + ' ' + str(draw) + '.csv'))
            age_structures_df_F.to_csv(
                os.path.join(age_structures_f_path, 'age_structures_df_F.csv' + ' ' + str(draw) + '.csv'))


if __name__ == '__main__':
    # Configuration
    num_draws = 1000
    dist_types = ['uniform', 'pert', 'beta', 'normal']  # available distributions

    # Load data
    data = load_all_data()
    blank_rows = pd.read_csv('../data/blank_rows.csv', encoding='utf-8')
    conteos = pd.concat([blank_rows, data['conteos']], axis=0, ignore_index=True)
    conteos = conteos[conteos['EDAD'].notnull()].copy()

    # Prepare tasks: deterministic and stochastic across all distributions
    tasks = [
        ('mid', None, 'mid_omissions'),
        ('low', None, 'low_omissions'),
        ('high', None, 'high_omissions'),
    ]
    for dist in dist_types:
        for i in range(num_draws):
            label = f'{dist}_draw_{i}'
            tasks.append(('draw', dist, label))

    def _execute_task(args):
        sample_type, dist, label = args
        df = conteos.copy()
        df['VALOR_corrected'] = correct_valor_for_omission(df, sample_type, distribution=dist)
        if dist is None:
            main_wrapper(df, label)
        else:
            main_wrapper(df, 'draw', dist, label)
        return label

    # Parallelize tasks loop using multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        for label in tqdm(pool.imap_unordered(_execute_task, tasks), total=len(tasks), desc='Tasks'):
            pass  # tasks run and log internally