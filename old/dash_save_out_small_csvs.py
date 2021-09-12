import numpy as np
import os
import pandas as pd

##
# -------- Load data
# NEW RR0/RR20 RANGES
# -- spatial, integral, VC and GM, EIR = 30, new rr20 range
# wi_name = 'spatialinside_integral2l4a_VC_and_GM_aEIR30_sweep_rc_d1_rr20_se2_newrr20'
# wi_name_sh = 'spatial, integral drive, VC and GM, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\a71\\ddc\\069\\a71ddc06-960b-ec11-a9ed-b88303911bc1'

# -- spatial, classic, VC and GM, EIR = 30, new rr0 range
# wi_name = 'spatialinside_classic3allele_VC_and_GM_aEIR30_sweep_rc_d_rr0_sne_newrr0'
# wi_name_sh = 'spatial, classic drive, VC and GM, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\8b9\\2f1\\679\\8b92f167-950b-ec11-a9ed-b88303911bc1'

# -- spatial, integral, GM only, EIR = 30, new rr20 range
# wi_name = 'spatialinside_integral2l4a_GM_only_aEIR30_sweep_rc_d1_rr20_se2_newrr20'
# wi_name_sh = 'spatial, integral drive, GM only, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\0dd\\3ce\\329\\0dd3ce32-960b-ec11-a9ed-b88303911bc1'

# -- spatial, classic, GM only, EIR = 30, new rr0 range
# wi_name = 'spatialinside_classic3allele_GM_only_aEIR30_sweep_rc_d_rr0_sne_newrr0'
# wi_name_sh = 'spatial, classic drive, GM only, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\89d\\46b\\9f9\\89d46b9f-950b-ec11-a9ed-b88303911bc1'

# ORIGINAL RUNS
# -- spatial, integral, VC and GM, EIR = 30
# wi_name = 'spatialinside_integral2l4a_VC_and_GM_aEIR30_sweep_rc_d1_rr20_se2'
# wi_name_sh = 'spatial, integral drive, VC and GM, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\789\\292\\b25\\789292b2-5505-ec11-a9ed-b88303911bc1'

# -- spatial, classic, VC and GM, EIR = 30
# wi_name = 'spatialinside_classic3allele_VC_and_GM_aEIR30_sweep_rc_d_rr0_sne'
# wi_name_sh = 'spatial, classic drive, VC and GM, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\a82\\f7d\\335\\a82f7d33-5705-ec11-a9ed-b88303911bc1'

# -- spatial, integral, GM only, EIR = 30
# wi_name = 'spatialinside_integral2l4a_GM_only_aEIR30_sweep_rc_d1_rr20_se2'
# wi_name_sh = 'spatial, integral drive, GM only, EIR = 30'
# data_dir = 'Z:\\home\\sleung\\workitems\\bf3\\d9c\\256\\bf3d9c25-6b04-ec11-a9ed-b88303911bc1'

# -- spatial, classic, GM only, EIR = 30
wi_name = 'spatialinside_classic3allele_GM_only_aEIR30_sweep_rc_d_rr0_sne_release_day_release_node_num'
wi_name_sh = 'spatial, classic drive, GM only, EIR = 30'
data_dir = 'Z:\\home\\sleung\\workitems\\648\\d61\\287\\648d6128-78f9-eb11-a9ed-b88303911bc1'

elim_day = 2555  # day on which elim fraction is calculated
num_sweep_vars = 6  # choose 4, 6
num_partition_vars = 3  # choose 0, 1, 3, 4
drive_type = 'classic'  # choose integral, classic

if num_sweep_vars == 6:
    if drive_type == 'classic':
        allvarvals_fns = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                          'd': [1, 0.95, 0.9],
                          'rr0': [0, 0.1, 0.2],
                          'sne': [0, 0.05, 0.1, 0.15, 0.2],
                          'release_day': [180, 240, 300, 360, 420, 480, 545],
                          'num_nodes': [6, 12]}
elif num_sweep_vars == 4:
    if drive_type == 'classic':
        allvarvals_fns = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                          'd': [1, 0.95, 0.9],
                          'rr0': [0, 0.1, 0.2],
                          'sne': [0, 0.05, 0.1, 0.15, 0.2]}
    elif drive_type == 'integral':
        allvarvals_fns = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                          'd1': [1, 0.95, 0.9],
                          'rr20': [0, 0.1, 0.2],
                          'se2': [0, 0.05, 0.1, 0.15, 0.2]}

if num_partition_vars > 0:
    if num_partition_vars == 1:
        partition_vars = ['d1']
        partition_vars_vals = [allvarvals_fns['d1']]
        file_suffix_ls = []
        for partition_vars_val0 in partition_vars_vals[0]:
            fsbegtemp = partition_vars[0] + str(partition_vars_val0)
            file_suffix_ls.append(fsbegtemp)
    elif num_partition_vars == 3:
        partition_vars = ['num_nodes', 'd', 'rr0']
        partition_vars_vals = [allvarvals_fns['num_nodes'], allvarvals_fns['d'], allvarvals_fns['rr0']]
        file_suffix_ls = []
        for partition_vars_val0 in partition_vars_vals[0]:
            fsbegtemp = partition_vars[0] + str(partition_vars_val0)
            for partition_vars_val1 in partition_vars_vals[1]:
                fsmidtemp = partition_vars[1] + str(partition_vars_val1)
                for partition_vars_val2 in partition_vars_vals[2]:
                    fsendtemp = partition_vars[2] + str(partition_vars_val2)
                    file_suffix_ls.append(fsbegtemp + '_' + fsmidtemp + '_' + fsendtemp)
    elif num_partition_vars == 4:
        partition_vars = ['num_nodes', 'd', 'rr0', 'sne']
        partition_vars_vals = [allvarvals_fns['num_nodes'], allvarvals_fns['d'],
                               allvarvals_fns['rr0'], allvarvals_fns['sne']]
        file_suffix_ls = []
        for partition_vars_val0 in partition_vars_vals[0]:
            fsbegtemp = partition_vars[0] + str(partition_vars_val0)
            for partition_vars_val1 in partition_vars_vals[1]:
                fsmidtemp = partition_vars[1] + str(partition_vars_val1)
                for partition_vars_val2 in partition_vars_vals[2]:
                    fsmid1temp = partition_vars[2] + str(partition_vars_val2)
                    for partition_vars_val3 in partition_vars_vals[3]:
                        fsendtemp = partition_vars[3] + str(partition_vars_val3)
                        file_suffix_ls.append(fsbegtemp + '_' + fsmidtemp + '_' + fsmid1temp + '_' + fsendtemp)
    dfi = pd.DataFrame()
    # dfa = pd.DataFrame()
    dfe = pd.DataFrame()
    dfed = pd.DataFrame()
    for file_suffix in file_suffix_ls:
        filei = os.path.join(data_dir, wi_name + '_inset_data_' + file_suffix + '.csv')
        filea = os.path.join(data_dir, wi_name + '_spatial_avg_allele_freqs_' + file_suffix + '.csv')
        filee = os.path.join(data_dir, wi_name + '_inset_data_elim_day_'
                             + str(elim_day) + '_indiv_sims_' + file_suffix + '.csv')
        fileed = os.path.join(data_dir, wi_name + '_inset_data_elim_day_number_indiv_sims_' + file_suffix + '.csv')
        dfi = dfi.append(pd.read_csv(filei))
        # dfa = dfa.append(pd.read_csv(filea))
        dfe = dfe.append(pd.read_csv(filee))
        dfed = dfed.append(pd.read_csv(fileed))
else:
    dfi = pd.read_csv(os.path.join(data_dir, wi_name + '_inset_data.csv'))
    # dfa = pd.read_csv(os.path.join(data_dir, wi_name + '_spatial_avg_allele_freqs.csv'))
    dfe = pd.read_csv(os.path.join(data_dir, wi_name + '_inset_data_elim_day_' + str(elim_day) + '_indiv_sims.csv'))
    dfed = pd.read_csv(os.path.join(data_dir, wi_name + '_inset_data_elim_day_number_indiv_sims.csv'))

# - Further clean up data
dfe = dfe.drop(columns=['Daily_EIR_elim', 'New_Clinical_Cases_elim', 'Run_Number'])
dfed = dfed.drop(columns=['Run_Number'])
if num_sweep_vars == 6:
    dfi.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)
    # dfa.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)
    dfe.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)
    dfed.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)
    if drive_type == 'classic':
        dfi = dfi[['Time', 'rc', 'd', 'rr0', 'sne', 'rd', 'nn',
                   'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std',
                   'True Prevalence', 'True Prevalence_std',
                   # 'Adult Vectors', 'Adult Vectors_std',
                   # 'Infectious Vectors', 'Infectious Vectors_std',
                   # 'Daily EIR', 'Daily EIR_std'
                   ]]
elif num_sweep_vars == 4:
    if drive_type == 'classic':
        dfi = dfi[['Time', 'rc', 'd', 'rr0', 'sne',
                   'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std',
                   'True Prevalence', 'True Prevalence_std',
                   # 'Adult Vectors', 'Adult Vectors_std',
                   # 'Infectious Vectors', 'Infectious Vectors_std',
                   # 'Daily EIR', 'Daily EIR_std'
                   ]]
    elif drive_type == 'integral':
        dfi = dfi[['Time', 'rc', 'd1', 'rr20', 'se2',
                   'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std',
                   'True Prevalence', 'True Prevalence_std',
                   # 'Adult Vectors', 'Adult Vectors_std',
                   # 'Infectious Vectors', 'Infectious Vectors_std',
                   # 'Daily EIR', 'Daily EIR_std'
                   ]]

dfi.to_csv('csvs/dfi_' + wi_name + '.csv', index=False)
# dfa.to_csv('csvs/dfa_' + wi_name + '.csv', index=False)
dfe.to_csv('csvs/dfe_' + wi_name + '.csv', index=False)
dfed.to_csv('csvs/dfed_' + wi_name + '.csv', index=False)
