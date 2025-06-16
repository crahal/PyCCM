import pandas as pd
import numpy as np

def make_projections(n, X, fert_start_idx,
                     conteos_all_2018_M_n_t,
                     conteos_all_2018_F_n_t,
                     lt_2018_F_t,
                     lt_2018_M_t,
                     conteos_all_2018_F_p_t,
                     conteos_all_2018_M_p_t,
                     asfr_2018,
                     l0):
    columns = [f't+{i}' for i in range(X + 1)]
    L_FF = np.zeros((n + 1, n + 1))
    L_MF = np.zeros((n + 1, n + 1))
    L_MM = np.zeros((n + 1, n + 1))
    SRB = conteos_all_2018_M_n_t.sum() / conteos_all_2018_F_n_t.sum()
    L0_F = lt_2018_F_t['Lx'].iloc[0]
    L0_M = lt_2018_M_t['Lx'].iloc[0]
    k_F = (1 / (1 + SRB)) * (L0_F / (2 * l0))
    k_M = (SRB / (1 + SRB)) * (L0_M / (2 * l0))
    ages = [int(label.split('-')[0]) for label in list(asfr_2018.index)]
    full_ages = lt_2018_F_t.index.astype(int)

    for i in range(len(ages) - 1):
        age = ages[i]
        next_age = ages[i + 1]
        col_idx = fert_start_idx + i
        F_j = asfr_2018['asfr'].iloc[i]
        F_next = asfr_2018['asfr'].iloc[i + 1]
        L_j = lt_2018_F_t['Lx'].loc[age]
        L_next = lt_2018_F_t['Lx'].loc[next_age]
        fertility_term = (F_j + F_next * (L_next / L_j)) * (L_j / L0_F)
        L_FF[0, col_idx] = k_F * fertility_term
        L_j = lt_2018_M_t['Lx'].loc[age]
        L_next = lt_2018_M_t['Lx'].loc[next_age]
        fertility_term = (F_j + F_next * (L_next / L_j)) * (L_j / L0_M)
        L_MF[0, col_idx] = k_M * fertility_term

    last_i = len(ages) - 1
    last_age = ages[last_i]
    last_col = fert_start_idx + last_i
    F_last = asfr_2018['asfr'].iloc[last_i]
    L_last = lt_2018_F_t['Lx'].loc[last_age]
    L_FF[0, last_col] = k_F * F_last * (L_last / L0_F)
    L_MF[0, last_col] = k_M * F_last * (L_last / L0_F)
    for i in range(1, n):
        L_FF[i, i - 1] = lt_2018_F_t['Lx'].values[i] / lt_2018_F_t['Lx'].values[i - 1]
        L_MM[i, i - 1] = lt_2018_M_t['Lx'].values[i] / lt_2018_M_t['Lx'].values[i - 1]
    L_FF[-1, -2] = lt_2018_F_t['Lx'].values[-1] / lt_2018_F_t['Lx'].values[-2]
    L_FF[-1, -1] = lt_2018_F_t["Tx"].values[-1] / lt_2018_F_t["Tx"].values[-2]
    L_MM[-1, -2] = lt_2018_M_t['Lx'].values[-1] / lt_2018_M_t['Lx'].values[-2]
    L_MM[-1, -1] = lt_2018_M_t["Tx"].values[-1] / lt_2018_M_t["Tx"].values[-2]

    n0_F = conteos_all_2018_F_p_t.copy()
    n0_M = conteos_all_2018_M_p_t.copy()
    n_proj_F = [n0_F]
    n_proj_M = [n0_M]

    for _ in range(X):
        n_proj_M.append((L_MF @ n_proj_F[-1]) + (L_MM @ n_proj_M[-1]))
        n_proj_F.append(L_FF @ n_proj_F[-1])

    age_structure_matrix_F = np.column_stack(n_proj_F)
    age_structures_df_F = pd.DataFrame(age_structure_matrix_F, index=full_ages, columns=columns)
    age_structure_matrix_M = np.column_stack(n_proj_M)
    age_structures_df_M = pd.DataFrame(age_structure_matrix_M, index=full_ages, columns=columns)

    return L_MM, L_MF, L_FF, age_structures_df_M, age_structures_df_F
