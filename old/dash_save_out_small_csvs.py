import numpy as np
import os
import pandas as pd

##
# -------- Set experiments/work items to load

# -- spatial, integral, VC and GM, EIR = 30
# wi_name = 'spatialinside_integral2l4a_VC_and_GM_aEIR30_sweep_rc_d1_rr20_se2'
# wi_name_sh = 'spatial, integral drive, VC and GM, EIR = 30'
# wi_names = ['spatialinside_integral2l4a_VC_and_GM_aEIR30_sweep_rc_d1_rr20_se2',
#               'spatialinside_integral2l4a_VC_and_GM_aEIR30_sweep_rc_d1_rr20_se2_newrr20',
#               'spatialinside_integral2l4a_VC_and_GM_aEIR30_sweep_rc_d1_rr20_se2_newse2']
# data_dirs = ['Y:\\home\\sleung\\workitems\\789\\292\\b25\\789292b2-5505-ec11-a9ed-b88303911bc1',
#              'Y:\\home\\sleung\\workitems\\a71\\ddc\\069\\a71ddc06-960b-ec11-a9ed-b88303911bc1',
#              'Y:\\home\\sleung\\workitems\\f63\\4e7\\327\\f634e732-7312-ec11-a9ed-b88303911bc1']
# num_sweep_vars = 4
# drive_type = 'integral'

# -- spatial, classic, VC and GM, EIR = 30
# wi_name = 'spatialinside_classic3allele_VC_and_GM_aEIR30_sweep_rc_d_rr0_sne'
# wi_name_sh = 'spatial, classic drive, VC and GM, EIR = 30'
# wi_names = ['spatialinside_classic3allele_VC_and_GM_aEIR30_sweep_rc_d_rr0_sne',
#             'spatialinside_classic3allele_VC_and_GM_aEIR30_sweep_rc_d_rr0_sne_newrr0',
#             'spatialinside_classic3allele_VC_and_GM_aEIR30_sweep_rc_d_rr0_sne_newsne']
# data_dirs = ['Y:\\home\\sleung\\workitems\\a82\\f7d\\335\\a82f7d33-5705-ec11-a9ed-b88303911bc1',
#              'Y:\\home\\sleung\\workitems\\8b9\\2f1\\679\\8b92f167-950b-ec11-a9ed-b88303911bc1',
#              'Y:\\home\\sleung\\workitems\\aba\\769\\f07\\aba769f0-7312-ec11-a9ed-b88303911bc1']
# num_sweep_vars = 4
# drive_type = 'classic'

# -- spatial, integral, GM only, EIR = 30
wi_name = 'spatialinside_integral2l4a_GM_only_aEIR30_sweep_rc_d1_rr20_se2'
wi_name_sh = 'spatial, integral drive, GM only, EIR = 30'
wi_names = ['spatialinside_integral2l4a_GM_only_aEIR30_sweep_rc_d1_rr20_se2',
            'spatialinside_integral2l4a_GM_only_aEIR30_sweep_rc_d1_rr20_se2_newrr20',
            'spatialinside_integral2l4a_GM_only_aEIR30_sweep_rc_d1_rr20_se2_newse2']
data_dirs = ['Z:\\home\\sleung\\workitems\\bf3\\d9c\\256\\bf3d9c25-6b04-ec11-a9ed-b88303911bc1',
             'Y:\\home\\sleung\\workitems\\0dd\\3ce\\329\\0dd3ce32-960b-ec11-a9ed-b88303911bc1',
             'Y:\\home\\sleung\\workitems\\63f\\310\\9f7\\63f3109f-7312-ec11-a9ed-b88303911bc1']
num_sweep_vars = 4
drive_type = 'integral'

# -- spatial, classic, GM only, EIR = 30
# NOTE THAT THIS WORK ITEM/EXP DOESN'T HAVE ALLELE FREQS
# wi_name = 'spatialinside_classic3allele_GM_only_aEIR30_sweep_rc_d_rr0_sne'
# wi_name_sh = 'spatial, classic drive, GM only, EIR = 30'
# wi_names = ['spatialinside_classic3allele_GM_only_aEIR30_sweep_rc_d_rr0_sne_release_day_release_node_num',
#             'spatialinside_classic3allele_GM_only_aEIR30_sweep_rc_d_rr0_sne_newrr0',
#             'spatialinside_classic3allele_GM_only_aEIR30_sweep_rc_d_rr0_sne_newsne']
# data_dirs = ['Z:\\home\\sleung\\workitems\\648\\d61\\287\\648d6128-78f9-eb11-a9ed-b88303911bc1',
#              'Y:\\home\\sleung\\workitems\\89d\\46b\\9f9\\89d46b9f-950b-ec11-a9ed-b88303911bc1',
#              'Y:\\home\\sleung\\workitems\\279\\e5f\\038\\279e5f03-8b12-ec11-a9ed-b88303911bc1']
# num_sweep_vars = 6
# drive_type = 'classic'

elim_day = 2555  # day on which elim fraction is calculated

file_suffixes = []
if (wi_name == 'spatialinside_integral2l4a_VC_and_GM_aEIR30_sweep_rc_d1_rr20_se2') \
        or (wi_name == 'spatialinside_integral2l4a_GM_only_aEIR30_sweep_rc_d1_rr20_se2'):
    for i in range(3):
        file_suffixes.append([])
    # - 1st work item
    partition_vars = ['d1']
    partition_vars_vals = [[1, 0.95, 0.9]]
    for partition_vars_val0 in partition_vars_vals[0]:
        fsbegtemp = partition_vars[0] + str(partition_vars_val0)
        file_suffixes[0].append(fsbegtemp)
    # - 2nd and 3rd work items have no partition vars
elif wi_name == 'spatialinside_classic3allele_VC_and_GM_aEIR30_sweep_rc_d_rr0_sne':
    # - 1st, 2nd, and 3rd work items have no partition vars
    for i in range(3):
        file_suffixes.append([])
elif wi_name == 'spatialinside_classic3allele_GM_only_aEIR30_sweep_rc_d_rr0_sne':
    for i in range(3):
        file_suffixes.append([])
    # - 1st work item
    partition_vars = ['num_nodes', 'd', 'rr0']
    partition_vars_vals = [[12, 6], [1, 0.95, 0.9], [0, 0.1, 0.2]]
    for partition_vars_val0 in partition_vars_vals[0]:
        fsbegtemp = partition_vars[0] + str(partition_vars_val0)
        for partition_vars_val1 in partition_vars_vals[1]:
            fsmidtemp = partition_vars[1] + str(partition_vars_val1)
            for partition_vars_val2 in partition_vars_vals[2]:
                fsendtemp = partition_vars[2] + str(partition_vars_val2)
                file_suffixes[0].append(fsbegtemp + '_' + fsmidtemp + '_' + fsendtemp)
    # - 2nd and 3rd work items have no partition vars

# -------- Load data
dfi = pd.DataFrame()
dfa = pd.DataFrame()
dfe = pd.DataFrame()
dfed = pd.DataFrame()
for iwn in range(len(file_suffixes)):
    file_suffixesnow = file_suffixes[iwn]
    data_dirnow = data_dirs[iwn]
    wi_namenow = wi_names[iwn]
    if len(file_suffixesnow) > 0:
        for file_suffix in file_suffixesnow:
            filei = os.path.join(data_dirnow, wi_namenow + '_inset_data_' + file_suffix + '.csv')
            filea = os.path.join(data_dirnow, wi_namenow + '_spatial_avg_allele_freqs_' + file_suffix + '.csv')
            filee = os.path.join(data_dirnow, wi_namenow + '_inset_data_elim_day_'
                                 + str(elim_day) + '_indiv_sims_' + file_suffix + '.csv')
            fileed = os.path.join(data_dirnow, wi_namenow + '_inset_data_elim_day_number_indiv_sims_' + file_suffix + '.csv')
            dfi = dfi.append(pd.read_csv(filei))
            dfa = dfa.append(pd.read_csv(filea))
            dfe = dfe.append(pd.read_csv(filee))
            dfed = dfed.append(pd.read_csv(fileed))
    else:
        dfi = dfi.append(pd.read_csv(os.path.join(data_dirnow, wi_namenow + '_inset_data.csv')))
        dfa = dfa.append(pd.read_csv(os.path.join(data_dirnow, wi_namenow + '_spatial_avg_allele_freqs.csv')))
        dfe = dfe.append(pd.read_csv(os.path.join(data_dirnow, wi_namenow + '_inset_data_elim_day_' + str(elim_day) + '_indiv_sims.csv')))
        dfed = dfed.append(pd.read_csv(os.path.join(data_dirnow, wi_namenow + '_inset_data_elim_day_number_indiv_sims.csv')))

# -------- Clean up data
dfe = dfe.drop(columns=['Daily_EIR_elim', 'New_Clinical_Cases_elim', 'Run_Number'])
dfed = dfed.drop(columns=['Run_Number'])
if drive_type == 'integral':
    dfi = dfi[dfi['rr20'] != 0.2]
    dfa = dfa[dfa['rr20'] != 0.2]
    dfe = dfe[dfe['rr20'] != 0.2]
    dfed = dfed[dfed['rr20'] != 0.2]
    if num_sweep_vars == 4:
        dfi = dfi[['Time', 'rc', 'd1', 'rr20', 'se2',
                   'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std',
                   'True Prevalence', 'True Prevalence_std',
                   # 'Adult Vectors', 'Adult Vectors_std',
                   # 'Infectious Vectors', 'Infectious Vectors_std',
                   # 'Daily EIR', 'Daily EIR_std'
                   ]]
elif drive_type == 'classic':
    dfi = dfi[dfi['rr0'] != 0.2]
    dfa = dfa[dfa['rr0'] != 0.2]
    dfe = dfe[dfe['rr0'] != 0.2]
    dfed = dfed[dfed['rr0'] != 0.2]
    if num_sweep_vars == 6:
        dfi.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)
        dfa.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)
        dfe.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)
        dfed.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)
        dfi = dfi[['Time', 'rc', 'd', 'rr0', 'sne', 'rd', 'nn',
                   'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std',
                   'True Prevalence', 'True Prevalence_std',
                   # 'Adult Vectors', 'Adult Vectors_std',
                   # 'Infectious Vectors', 'Infectious Vectors_std',
                   # 'Daily EIR', 'Daily EIR_std'
                   ]]
    elif num_sweep_vars == 4:
        dfi = dfi[['Time', 'rc', 'd', 'rr0', 'sne',
                   'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std',
                   'True Prevalence', 'True Prevalence_std',
                   # 'Adult Vectors', 'Adult Vectors_std',
                   # 'Infectious Vectors', 'Infectious Vectors_std',
                   # 'Daily EIR', 'Daily EIR_std'
                   ]]

# -------- Save out data
dfi.to_csv('csvs/dfi_' + wi_name + '.csv', index=False)
dfa.to_csv('csvs/dfa_' + wi_name + '.csv', index=False)
dfe.to_csv('csvs/dfe_' + wi_name + '.csv', index=False)
dfed.to_csv('csvs/dfed_' + wi_name + '.csv', index=False)
