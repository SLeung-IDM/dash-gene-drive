import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

## TO DO:
# add option to choose spatial/single node, classic/integral, GM only/VC+GM, EIR
# for supp figs, that'd be...
# eir = 10: (2 x 2 x 1 x 1) = 4
# eir = 30: (2 x 2 x 2 x 1) = 8
# eir = 80: (2 x 2 x 1 x 1) = 4
# --> 4 + 8 + 4 = 16 supplementary figures

##
# -------- Load data

# NEW RC X RELEASE_NUMBER RUNS
# -- spatial, integral, VC and GM, EIR = 30
# wi_name = 'spatialinside_integral2l4a_VC_and_GM_aEIR30_sweep_rc_release_number'
# wi_name_sh = 'spatial, integral drive, VC and GM, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\85c\\f99\\239\\85cf9923-920b-ec11-a9ed-b88303911bc1'

# -- spatial, classic, VC and GM, EIR = 30
# wi_name = 'spatialinside_classic3allele_VC_and_GM_aEIR30_sweep_rc_release_number'
# wi_name_sh = 'spatial, classic drive, VC and GM, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\d54\\92b\\747\\d5492b74-7f0b-ec11-a9ed-b88303911bc1'

# -- spatial, integral, GM only, EIR = 30
# wi_name = 'spatialinside_integral2l4a_GM_only_aEIR30_sweep_release_number'
# wi_name_sh = 'spatial, integral drive, GM only, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\52f\\74b\\719\\52f74b71-920b-ec11-a9ed-b88303911bc1'

# -- spatial, classic, GM only, EIR = 30
# wi_name = 'spatialinside_classic3allele_GM_only_aEIR30_sweep_release_number'
# wi_name_sh = 'spatial, classic drive, GM only, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\23b\\ec6\\eb9\\23bec6eb-910b-ec11-a9ed-b88303911bc1'


# NEW RC X SNE/SE2 RUNS
# -- spatial, integral, VC and GM, EIR = 30
# wi_name = 'spatialinside_integral2l4a_VC_and_GM_aEIR30_sweep_rc_se2_newse2'
# wi_name_sh = 'spatial, integral drive, VC and GM, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\7c9\\5dd\\089\\7c95dd08-940b-ec11-a9ed-b88303911bc1'

# -- spatial, classic, VC and GM, EIR = 30
# wi_name = 'spatialinside_classic3allele_VC_and_GM_aEIR30_sweep_rc_sne_newsne'
# wi_name_sh = 'spatial, classic drive, VC and GM, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\fa7\\74c\\769\\fa774c76-940b-ec11-a9ed-b88303911bc1'

# -- spatial, integral, GM only, EIR = 30
# wi_name = 'spatialinside_integral2l4a_GM_only_aEIR30_sweep_rc_se2_newse2'
# wi_name_sh = 'spatial, integral drive, GM only, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\218\\bf8\\1f9\\218bf81f-930b-ec11-a9ed-b88303911bc1'

# -- spatial, classic, GM only, EIR = 30
# wi_name = 'spatialinside_classic3allele_GM_only_aEIR30_sweep_rc_sne_newsne'
# wi_name_sh = 'spatial, classic drive, GM only, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\203\\bea\\d89\\203bead8-940b-ec11-a9ed-b88303911bc1'


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
# -- spatial, integral, VC and GM, EIR = 80
# wi_name = 'spatialinside_integral2l4a_VC_and_GM_aEIR80_sweep_rc_d1_rr20_se2'
# wi_name_sh = 'spatial, integral drive, VC and GM, EIR = 80'
# data_dir = 'Y:\\home\\sleung\\workitems\\41d\\361\\795\\41d36179-5605-ec11-a9ed-b88303911bc1'

# -- spatial, classic, VC and GM, EIR = 80
# wi_name = 'spatialinside_classic3allele_VC_and_GM_aEIR80_sweep_rc_d_rr0_sne_release_day_release_node_num'
# wi_name_sh = 'spatial, classic drive, VC and GM, EIR = 80'
# data_dir = 'Z:\\home\\sleung\\workitems\\5de\\cc6\\036\\5decc603-6a04-ec11-a9ed-b88303911bc1'

# -- spatial, integral, VC and GM, EIR = 30
# wi_name = 'spatialinside_integral2l4a_VC_and_GM_aEIR30_sweep_rc_d1_rr20_se2'
# wi_name_sh = 'spatial, integral drive, VC and GM, EIR = 30'
# data_dir = 'Y:\\home\\sleung\\workitems\\789\\292\\b25\\789292b2-5505-ec11-a9ed-b88303911bc1'

# -- spatial, classic, VC and GM, EIR = 30
wi_name = 'spatialinside_classic3allele_VC_and_GM_aEIR30_sweep_rc_d_rr0_sne'
wi_name_sh = 'spatial, classic drive, VC and GM, EIR = 30'
data_dir = 'Y:\\home\\sleung\\workitems\\a82\\f7d\\335\\a82f7d33-5705-ec11-a9ed-b88303911bc1'

# -- spatial, integral, GM only, EIR = 30
# wi_name = 'spatialinside_integral2l4a_GM_only_aEIR30_sweep_rc_d1_rr20_se2'
# wi_name_sh = 'spatial, integral drive, GM only, EIR = 30'
# data_dir = 'Z:\\home\\sleung\\workitems\\bf3\\d9c\\256\\bf3d9c25-6b04-ec11-a9ed-b88303911bc1'

# -- spatial, classic, GM only, EIR = 30
# wi_name = 'spatialinside_classic3allele_GM_only_aEIR30_sweep_rc_d_rr0_sne_release_day_release_node_num'
# wi_name_sh = 'spatial, classic drive, GM only, EIR = 30'
# data_dir = 'Z:\\home\\sleung\\workitems\\648\\d61\\287\\648d6128-78f9-eb11-a9ed-b88303911bc1'

# -- spatial, integral, VC and GM, EIR = 10
# wi_name = 'spatialinside_integral2l4a_VC_and_GM_aEIR10_sweep_rc_d1_rr20_se2'
# wi_name_sh = 'spatial, integral drive, VC and GM, EIR = 10'
# data_dir = 'Y:\\home\\sleung\\workitems\\827\\ee8\\3d5\\827ee83d-5605-ec11-a9ed-b88303911bc1'

# -- spatial, classic, VC and GM, EIR = 10
# wi_name = 'spatialinside_classic3allele_VC_and_GM_aEIR10_sweep_rc_d_rr0_sne'
# wi_name_sh = 'spatial, classic drive, VC and GM, EIR = 10'
# data_dir = 'Y:\\home\\sleung\\workitems\\ba2\\a75\\a15\\ba2a75a1-5705-ec11-a9ed-b88303911bc1'

# -- spatial, integral, GM only, EIR = 10
# wi_name = 'spatialinside_integral2l4a_GM_only_aEIR10_sweep_rc_d1_rr20_se2'
# wi_name_sh = 'spatial, integral drive, GM only, EIR = 10'
# data_dir = 'Y:\\home\\sleung\\workitems\\cd0\\917\\d95\\cd0917d9-5205-ec11-a9ed-b88303911bc1'

# -- spatial, classic, GM only, EIR = 10
# wi_name = 'spatialinside_classic3allele_GM_only_aEIR10_sweep_rc_d_rr0_sne_release_day_release_node_num'
# wi_name_sh = 'spatial, classic drive, GM only, EIR = 10'
# data_dir = 'Z:\\home\\sleung\\workitems\\d2b\\2a2\\f47\\d2b2a2f4-77f9-eb11-a9ed-b88303911bc1'

num_sweep_vars = 4  # choose 4, 6
num_partition_vars = 0  # choose 0, 1, 3, 4
drive_type = 'classic'  # choose integral, classic

distrib_itns = True
if distrib_itns == True:
    itn_distrib_days = [180, 3 * 365 + 180, 6 * 365 + 180]

released_mosqs = True
if released_mosqs == True:
    released_day = 180

num_yrs = 8  # length of sim
elim_day = 2555  # day on which elim fraction is calculated
af_hm_day = 365 * 8 - 1  # day on which end of sim allele freq is calculated
num_seeds = 20  # num of seeds per sim

if num_sweep_vars == 6:
    if drive_type == 'classic':
        allvardefs = {'rc': 1, 'd': 1, 'rr0': 0, 'sne': 0,
                      'rd': 180, 'nn': 6}
        allvarvals = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                      'd': [1, 0.95, 0.9],
                      'rr0': [0, 0.1, 0.2],
                      'sne': [0, 0.05, 0.1, 0.15, 0.2],
                      'rd': [180, 240, 300, 360, 420, 480, 545],
                      'nn': [6, 12]}
        allvarvals_fns = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                          'd': [1, 0.95, 0.9],
                          'rr0': [0, 0.1, 0.2],
                          'sne': [0, 0.05, 0.1, 0.15, 0.2],
                          'release_day': [180, 240, 300, 360, 420, 480, 545],
                          'num_nodes': [6, 12]}
elif num_sweep_vars == 4:
    if drive_type == 'classic':
        # NEW RC X RELEASE_NUMBER
        # allvardefs = {
        #     'rc': 0.6,  # VC and GM
        #     # 'rc': 0.8,  # GM only
        #     'd': 0.95,
        #     'sne': 0.1,
        #     'rr0': 0.01}
        # allvarvals = {
        #     'rc': [0.7, 0.6],  # VC and GM
        #     # 'rc': [0.8],  # GM only
        #     'd': [0.95],
        #     'rr0': [0.01],
        #     'sne': [0.1],
        #     'release_number': [1000, 10000]
        # }
        # NEW RC X SNE
        # allvardefs = {'rc': 1, 'd': 0.95, 'sne': 0.25, 'rr0': 0.01}
        # allvarvals = {
        #     'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
        #     # 'rc': [1, 0.9, 0.8],
        #     'd': [0.95],
        #     'rr0': [0.01],
        #     'sne': [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]}
        # OLD
        allvardefs = {'rc': 1, 'd': 1, 'sne': 0,
                      'rr0': 0}
                      # 'rr0': 0.01}  # NEW RR0
        allvarvals = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                      'd': [1, 0.95, 0.9],
                      'rr0': [0, 0.1, 0.2],
                      # 'rr0': [0.001, 0.01],  # NEW RR0
                      'sne': [0, 0.05, 0.1, 0.15, 0.2]}
        allvarvals_fns = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                          'd': [1, 0.95, 0.9],
                          'rr0': [0, 0.1, 0.2],
                          'sne': [0, 0.05, 0.1, 0.15, 0.2]}
    elif drive_type == 'integral':
        # NEW RC X RELEASE_NUMBER
        # allvardefs = {
        #     'rc': 0.6,  # VC and GM
        #     # 'rc': 0.8,  # GM only
        #     'd1': 0.95,
        #     'se2': 0.1,
        #     'rr20': 0.01}
        # allvarvals = {
        #     'rc': [0.7, 0.6],  # VC and GM
        #     # 'rc': [0.8],  # GM only
        #     'd1': [0.95],
        #     'rr20': [0.01],
        #     'se2': [0.1],
        #     'release_number': [1000, 10000]
        # }
        # NEW RC X SE2
        # allvardefs = {'rc': 1, 'd1': 0.95, 'se2': 0.25, 'rr20': 0.01}
        # allvarvals = {
        #     'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
        #     # 'rc': [1, 0.9, 0.8],
        #     'd1': [0.95],
        #     'rr20': [0.01],
        #     'se2': [0.25, 0.3, 0.35, 0.4, 0.45, 0.4]
        # }
        # OLD
        allvardefs = {'rc': 1, 'd1': 1, 'se2': 0,
                      # 'rr20': 0}
                      'rr20': 0.01}  # NEW RR20
        allvarvals = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                      'd1': [1, 0.95, 0.9],
                      # 'rr20': [0, 0.1, 0.2],
                      'rr20': [0.001, 0.01],  # NEW RR20
                      'se2': [0, 0.05, 0.1, 0.15, 0.2]}
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
        # filea = os.path.join(data_dir, wi_name + '_spatial_avg_allele_freqs_' + file_suffix + '.csv')
        filee = os.path.join(data_dir, wi_name + '_inset_data_elim_day_'
                             + str(elim_day) + '_indiv_sims_' + file_suffix + '.csv')
        fileed = os.path.join(data_dir, wi_name + '_inset_data_elim_day_number_indiv_sims_' + file_suffix + '.csv')
        dfi = dfi.append(pd.read_csv(filei))
        # dfa = dfa.append(pd.read_csv(filea))
        dfe = dfe.append(pd.read_csv(filee))
        dfed = dfed.append(pd.read_csv(fileed))
else:
    dfi = pd.read_csv(os.path.join(data_dir, wi_name + '_inset_data.csv'))
    dfa = pd.read_csv(os.path.join(data_dir, wi_name + '_spatial_avg_allele_freqs.csv'))
    dfe = pd.read_csv(os.path.join(data_dir, wi_name + '_inset_data_elim_day_' + str(elim_day) + '_indiv_sims.csv'))
    dfed = pd.read_csv(os.path.join(data_dir, wi_name + '_inset_data_elim_day_number_indiv_sims.csv'))

# - Clean up data
# dfa.rename(columns={'Time': 'time'}, inplace=True)
dfe.rename(columns={'Time': 'time'}, inplace=True)
dfed.rename(columns={'Time': 'time'}, inplace=True)

# TEMPORARY ADDITIONS TO DATAFRAMES
# NEW RC X RELEASE_NUMBER
# if drive_type == 'integral':
#     dfi['rr20'] = 0.01
#     dfi['d1'] = 0.95
#     dfi['se2'] = 0.1
#     # dfi['rc'] = 0.8  # GM only
#     dfe['rr20'] = 0.01
#     dfe['d1'] = 0.95
#     dfe['se2'] = 0.1
#     # dfe['rc'] = 0.8  # GM only
#     dfed['rr20'] = 0.01
#     dfed['d1'] = 0.95
#     dfed['se2'] = 0.1
#     # dfed['rc'] = 0.8  # GM only
# elif drive_type == 'classic':
#     dfi['rr0'] = 0.01
#     dfi['d'] = 0.95
#     dfi['sne'] = 0.1
#     # dfi['rc'] = 0.8  # GM only
#     dfe['rr0'] = 0.01
#     dfe['d'] = 0.95
#     dfe['sne'] = 0.1
#     # dfe['rc'] = 0.8  # GM only
#     dfed['rr0'] = 0.01
#     dfed['d'] = 0.95
#     dfed['sne'] = 0.1
#     # dfed['rc'] = 0.8  # GM only
# NEW RC X SNE/SE2
# if drive_type == 'integral':
#     dfi['rr20'] = 0.01
#     dfi['d1'] = 0.95
#     dfe['rr20'] = 0.01
#     dfe['d1'] = 0.95
#     dfed['rr20'] = 0.01
#     dfed['d1'] = 0.95
# elif drive_type == 'classic':
#     dfi['rr0'] = 0.01
#     dfi['d'] = 0.95
#     dfe['rr0'] = 0.01
#     dfe['d'] = 0.95
#     dfed['rr0'] = 0.01
#     dfed['d'] = 0.95

# - Further clean up data
dfi.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)
# dfa.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)
dfe = dfe.drop(columns=['Daily_EIR_elim', 'New_Clinical_Cases_elim', 'Run_Number'])
dfe.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)
dfed.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)
if num_sweep_vars == 6:
    if drive_type == 'classic':
        dfi = dfi[['Time', 'rc', 'd', 'rr0', 'sne', 'rd', 'nn', 'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std']]
elif num_sweep_vars == 4:
    # NEW RC X RELEASE_NUMBER
    # if drive_type == 'classic':
    #     dfi = dfi[['Time', 'release_number', 'rc', 'd', 'rr0', 'sne', 'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std']]
    # elif drive_type == 'integral':
    #     dfi = dfi[['Time', 'release_number', 'rc', 'd1', 'rr20', 'se2', 'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std']]
    # OLD
    if drive_type == 'classic':
        dfi = dfi[['Time', 'rc', 'd', 'rr0', 'sne', 'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std']]
    elif drive_type == 'integral':
        dfi = dfi[['Time', 'rc', 'd1', 'rr20', 'se2', 'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std']]

# dfp = pd.read_csv('prev.csv')
# dfp.rename(columns={'time': 'Time'}, inplace=True)

##
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Elimination probabilities', children=[

            html.H2(children='Elim probabilities: ' + wi_name_sh),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='outer-xvar0',
                        options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                        value='rc'
                    )
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='outer-yvar0',
                        options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                        value='d'
                    )
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Matrix x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='matrix-xvar0',
                        options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                        value='rr0')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Matrix y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='matrix-yvar0',
                        options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                        value='sne')
                ], style={'width': '10%'}),

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='elim-prob-matrices',
                          style={'width': '95%', 'height': '80vh'})
            ])
        ]),

        dcc.Tab(label='Years to elimination', children=[

            html.H2(children='Years to elim: ' + wi_name_sh),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='outer-xvar1',
                        options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                        value='rc'
                    )
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='outer-yvar1',
                        options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                        value='d'
                    )
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Matrix x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='matrix-xvar1',
                        options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                        value='rr0')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Matrix y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='matrix-yvar1',
                        options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                        value='sne')
                ], style={'width': '10%'}),

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='elim-day-matrices',
                          style={'width': '95%', 'height': '80vh'})
            ])
        ]),

        dcc.Tab(label='Prevalence time series', children=[

            html.H2(children='Prev time series: ' + wi_name_sh),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='outer-xvar2',
                        options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                        value='rr0'
                    )
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='outer-yvar2',
                        options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                        value='sne'
                    )
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['1st sweep var (color):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='sweep-var2-0',
                        options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                        value='rc')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['2nd sweep var (line style):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='sweep-var2-1',
                        options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                        value='d')
                ], style={'width': '10%'}),

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='prev-ts',
                          style={'width': '100%', 'height': '80vh'})
            ])
        ])
    ])

])


@app.callback(
    Output('elim-prob-matrices', 'figure'),
    [Input('outer-xvar0', 'value'),
     Input('outer-yvar0', 'value'),
     Input('matrix-xvar0', 'value'),
     Input('matrix-yvar0', 'value')])
def update_elim_prob_matrices(ov_xvar, ov_yvar, mat_xvar, mat_yvar):
    # - Get all outer xvar and yvar vals
    ov_xvar_vals = allvarvals[ov_xvar]
    ov_yvar_vals = allvarvals[ov_yvar]

    # - Compute subplot titles and heatmaps
    iaxis = 1
    subplots = []
    subplot_titles = []

    for ov_yvar_val in ov_yvar_vals:
        for ov_xvar_val in ov_xvar_vals:

            # - Compute heatmap
            allvardefsnow = {k: v for k, v in allvardefs.items() if k not in [mat_xvar, mat_yvar, ov_xvar, ov_yvar]}
            dfenow = dfe
            for k, v in allvardefsnow.items():
                dfenow = dfenow[dfenow[k] == v]
                dfenow.drop(columns=[k], inplace=True)
            dfenow = dfenow[dfenow[ov_xvar] == ov_xvar_val]
            dfenow = dfenow[dfenow[ov_yvar] == ov_yvar_val]
            dfenow.drop(columns=[ov_xvar, ov_yvar], inplace=True)
            dfenownow = (dfenow.groupby([mat_xvar, mat_yvar])['True_Prevalence_elim'].sum() / num_seeds).reset_index()
            matnow = dfenownow.pivot_table(index=[mat_yvar], columns=[mat_xvar], values='True_Prevalence_elim')

            # - Create annotated heatmap
            subplots.append(ff.create_annotated_heatmap(
                z=matnow.values,
                x=matnow.columns.tolist(),
                y=matnow.index.tolist(),
                zmax=1,
                zmin=0,
                # coloraxis='coloraxis',
                # hovertemplate=mat_xvar + ': %{x}<br>' + mat_yvar + ': %{y}<br>Elim prob: %{z}<extra></extra>',
                showscale=True,
                colorscale='YlOrBr_r')
            )

            # - Update annotation axes
            for annot in subplots[-1]['layout']['annotations']:
                annot['xref'] = 'x' + str(iaxis)
                annot['yref'] = 'y' + str(iaxis)
            iaxis = iaxis + 1

            # - Create subplot titles
            # subplot_titles.append(ov_xvar + '=' + str(ov_xvar_val) + ', '
            #                      + ov_yvar + '=' + str(ov_yvar_val))

    # - Set up subplot framework and titles
    fig = make_subplots(
        rows=len(ov_yvar_vals), cols=len(ov_xvar_vals),
        # subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        column_titles=[ov_xvar + '=' + str(val) for val in ov_xvar_vals],
        row_titles=[ov_yvar + '=' + str(val) for val in ov_yvar_vals],
        x_title=mat_xvar,
        y_title=mat_yvar,
        horizontal_spacing=0.03,
        vertical_spacing=0.03
    )

    # - Create each subplot
    isp = 0
    for irow, ov_yvar_val in enumerate(ov_yvar_vals):
        for icol, ov_xvar_val in enumerate(ov_xvar_vals):
            fig.add_trace(subplots[isp].data[0], row=irow + 1, col=icol + 1)
            isp = isp + 1

    # - Update annotations for all subplot
    for isp, subplot in enumerate(subplots):
        fig.layout.annotations += subplots[isp].layout.annotations

    # - Update fig layout and subplot axes
    fig.update_xaxes(
        tickmode='array',
        tickvals=allvarvals[mat_xvar],
        ticktext=[str(val) for val in allvarvals[mat_xvar]]
    )
    fig.update_yaxes(
        tickmode='array',
        tickvals=allvarvals[mat_yvar],
        ticktext=[str(val) for val in allvarvals[mat_yvar]]
    )
    # fig.update_coloraxes(colorscale='Viridis')
    fig.update_layout(margin=dict(l=60, r=50, b=50, t=30))
    #                   coloraxis={'colorscale': 'YlOrBr_r'},
    #                   title='Elim probabilities, ' + wi_name,
    #                   transition_duration=500)

    return fig


@app.callback(
    Output('elim-day-matrices', 'figure'),
    [Input('outer-xvar1', 'value'),
     Input('outer-yvar1', 'value'),
     Input('matrix-xvar1', 'value'),
     Input('matrix-yvar1', 'value')])
def update_elim_day_matrices(ov_xvar, ov_yvar, mat_xvar, mat_yvar):
    # - Get all outer xvar and yvar vals
    ov_xvar_vals = allvarvals[ov_xvar]
    ov_yvar_vals = allvarvals[ov_yvar]

    # - Compute subplot titles and heatmaps
    iaxis = 1
    subplots = []

    for ov_yvar_val in ov_yvar_vals:
        for ov_xvar_val in ov_xvar_vals:

            # - Compute heatmap
            allvardefsnow = {k: v for k, v in allvardefs.items() if k not in [mat_xvar, mat_yvar, ov_xvar, ov_yvar]}
            dfednow = dfed
            for k, v in allvardefsnow.items():
                dfednow = dfednow[dfednow[k] == v]
                dfednow.drop(columns=[k], inplace=True)
            dfednow = dfednow[dfednow[ov_xvar] == ov_xvar_val]
            dfednow = dfednow[dfednow[ov_yvar] == ov_yvar_val]
            dfednow.drop(columns=[ov_xvar, ov_yvar], inplace=True)
            dfednow.loc[dfednow['True_Prevalence_elim'] == False,
                        'True_Prevalence_elim_day'] = np.nan
            dfednow.drop(columns=['True_Prevalence_elim'], inplace=True)
            dfednownow = (dfednow.groupby([mat_xvar, mat_yvar])['True_Prevalence_elim_day'].mean()).reset_index()
            matnow = dfednownow.pivot_table(index=[mat_yvar], columns=[mat_xvar],
                                            values='True_Prevalence_elim_day', dropna=False)
            # matnow = matnow.round(1)  # .astype('Int64')
            matnow = (matnow / 365).round(1)  # .astype('Int64')
            # z_text = [[str(y) for y in x] for x in matnow.values]

            # - Create annotated heatmap
            subplots.append(ff.create_annotated_heatmap(
                z=matnow.values,
                x=matnow.columns.tolist(),
                y=matnow.index.tolist(),
                # annotation_text=z_text,
                zmax=(dfed['True_Prevalence_elim_day'] / 365).max(),
                zmin=(dfed['True_Prevalence_elim_day'] / 365).min(),
                showscale=True,
                colorscale='YlOrBr')
            )

            # - Update annotation axes
            for annot in subplots[-1]['layout']['annotations']:
                annot['xref'] = 'x' + str(iaxis)
                annot['yref'] = 'y' + str(iaxis)
            iaxis = iaxis + 1

    # - Set up subplot framework and titles
    fig = make_subplots(
        rows=len(ov_yvar_vals), cols=len(ov_xvar_vals),
        shared_xaxes=True,
        shared_yaxes=True,
        column_titles=[ov_xvar + '=' + str(val) for val in ov_xvar_vals],
        row_titles=[ov_yvar + '=' + str(val) for val in ov_yvar_vals],
        x_title=mat_xvar,
        y_title=mat_yvar,
        horizontal_spacing=0.03,
        vertical_spacing=0.03
    )

    # - Create each subplot
    isp = 0
    for irow, ov_yvar_val in enumerate(ov_yvar_vals):
        for icol, ov_xvar_val in enumerate(ov_xvar_vals):
            fig.add_trace(subplots[isp].data[0], row=irow + 1, col=icol + 1)
            isp = isp + 1

    # - Update annotations for all subplot
    for isp, subplot in enumerate(subplots):
        fig.layout.annotations += subplots[isp].layout.annotations

    # - Update fig layout and subplot axes
    fig.update_xaxes(
        ticklen=10,
        tickmode='array',
        tickvals=allvarvals[mat_xvar],
        ticktext=[str(val) for val in allvarvals[mat_xvar]]
    )
    fig.update_yaxes(
        ticklen=10,
        tickmode='array',
        tickvals=allvarvals[mat_yvar],
        ticktext=[str(val) for val in allvarvals[mat_yvar]]
    )
    fig.update_layout(margin=dict(l=60, r=50, b=50, t=30))

    return fig


@app.callback(
    Output('prev-ts', 'figure'),
    [Input('outer-xvar2', 'value'),
     Input('outer-yvar2', 'value'),
     Input('sweep-var2-0', 'value'),
     Input('sweep-var2-1', 'value')])
def update_prev_ts(ov_xvar, ov_yvar, svar0, svar1):
    allvardefsnow = {k: v for k, v in allvardefs.items() if k not in [svar0, svar1, ov_xvar, ov_yvar]}
    dfinow = dfi
    for k, v in allvardefsnow.items():
        dfinow = dfinow[dfinow[k] == v]
        dfinow.drop(columns=[k], inplace=True)

    fig = px.line(dfinow, x='Time', y='PfHRP2 Prevalence',
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    return fig


##
if __name__ == '__main__':
    app.run_server(debug=False, port=8080)
