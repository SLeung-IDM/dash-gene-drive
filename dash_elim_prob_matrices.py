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

##
# -------- Load data
wi_name = 'spatialinside_classic3allele_GM_only_aEIR30_sweep_rc_d_rr0_sne_release_day_release_node_num'
data_dir = 'Z:\\home\\sleung\\workitems\\648\\d61\\287\\648d6128-78f9-eb11-a9ed-b88303911bc1'

num_yrs = 8  # length of sim
elim_day = 2555  # day on which elim fraction is calculated
af_hm_day = 365 * 8 - 1  # day on which end of sim allele freq is calculated
num_seeds = 20  # num of seeds per sim
drive_type = "classic"  # choose: classic, integral

released_mosqs = True
if released_mosqs == True:
    released_day = 180

distrib_itns = True
if distrib_itns == True:
    itn_distrib_days = [180, 3 * 365 + 180, 6 * 365 + 180]

allvardefs = {'rc': 1, 'd': 1, 'rr0': 0,
              'sne': 0, 'release_day': 180, 'num_nodes': 6}
allvarvals = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
              'd': [1, 0.95, 0.9],
              'rr0': [0, 0.1, 0.2],
              'sne': [0, 0.05, 0.1, 0.15, 0.2],
              'release_day': [180, 240, 300, 360, 420, 480, 545],
              'num_nodes': [6, 12]}

partition_vars = ['num_nodes', 'd', 'rr0']
partition_vars_vals = [allvarvals['num_nodes'], allvarvals['d'], allvarvals['rr0']]

file_suffix_ls = []

for partition_vars_val0 in partition_vars_vals[0]:
    fsbegtemp = partition_vars[0] + str(partition_vars_val0)
    for partition_vars_val1 in partition_vars_vals[1]:
        fsmidtemp = partition_vars[1] + str(partition_vars_val1)
        for partition_vars_val2 in partition_vars_vals[2]:
            fsendtemp = partition_vars[2] + str(partition_vars_val2)
            file_suffix_ls.append(fsbegtemp + '_' + fsmidtemp + '_' + fsendtemp)

# dfi = pd.DataFrame()
# dfa = pd.DataFrame()
dfe = pd.DataFrame()
# dfed = pd.DataFrame()
for file_suffix in file_suffix_ls:
    # filei = os.path.join(data_dir, wi_name + '_inset_data_' + file_suffix + '.csv')
    # filea = os.path.join(data_dir, wi_name + '_spatial_avg_allele_freqs_' + file_suffix + '.csv')
    filee = os.path.join(data_dir, wi_name + '_inset_data_elim_day_'
                         + str(elim_day) + '_indiv_sims_' + file_suffix + '.csv')
    # fileed = os.path.join(data_dir, wi_name + '_inset_data_elim_day_number_indiv_sims_' + file_suffix + '.csv')
    # dfi = dfi.append(pd.read_csv(filei))
    # dfa = dfa.append(pd.read_csv(filea))
    dfe = dfe.append(pd.read_csv(filee))
    # dfed = dfed.append(pd.read_csv(fileed))

# - Clean up data
# if 'Unnamed: 0' in dfi.columns:
#     dfi = dfi.drop('Unnamed: 0', axis=1)
# if 'Unnamed: 0' in dfa.columns:
#     dfa = dfa.drop('Unnamed: 0', axis=1)
if 'Unnamed: 0' in dfe.columns:
    dfe = dfe.drop('Unnamed: 0', axis=1)
# if 'Unnamed: 0' in dfed.columns:
#     dfed = dfed.drop('Unnamed: 0', axis=1)
# dfa.rename(columns={'Time': 'time'}, inplace=True)
dfe.rename(columns={'Time': 'time'}, inplace=True)
# dfed.rename(columns={'Time': 'time'}, inplace=True)

# - Further clean up data
dfesm = dfe.drop(columns=['Daily_EIR_elim', 'New_Clinical_Cases_elim', 'Run_Number'])

##
app.layout = html.Div([

    html.Div([
        dcc.Dropdown(
            id='matrix-xvar',
            options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
            value='rc')
    ], style={'width': '49%', 'padding': '0px 20px 20px 20px'}),

    html.Div([
        dcc.Dropdown(
            id='matrix-yvar',
            options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
            value='d')
    ], style={'width': '49%', 'padding': '0px 20px 20px 20px'}),

    html.Div([
        dcc.Slider(
            id='rc-slider',
            min=dfe['rc'].min(),
            max=dfe['rc'].max(),
            value=sv1_def,
            marks={str(rc): str(rc) for rc in dfe['rc'].unique()},
            step=None)
    ], style={'width': '49%', 'padding': '0px 20px 20px 20px'}),

    html.Div([
        dcc.Slider(
            id='d-slider',
            min=dfe['d'].min(),
            max=dfe['d'].max(),
            value=sv2_def,
            marks={str(d): str(d) for d in dfe['d'].unique()},
            step=None
        )
    ], style={'width': '49%', 'padding': '0px 20px 20px 20px'}),

    html.Div([
        dcc.Graph(id='elim-prob-matrix')
    ], style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


@app.callback(
    Output('elim-prob-matrix', 'figure'),
    [Input('x-axis-var', 'value'),
     Input('y-axis-var', 'value'),
     Input('rc-slider', 'value'),
     Input('d-slider', 'value')])
def update_figure(xvar, yvar, selected_rc, selected_d):
    allvars = {sv1_str: selected_rc, sv2_str: selected_d, sv3_str: sv3_def, sv4_str: sv4_def, sv5_str: sv5_def,
               sv6_str: sv6_def}
    allvars = {k: v for k, v in allvars.items() if k not in [xvar, yvar]}
    dfenow = dfe.drop(columns=['Daily_EIR_elim', 'New_Clinical_Cases_elim', 'Run_Number'])
    for k, v in allvars.items():
        dfenow = dfenow[dfenow[k] == v]
        dfenow.drop(columns=[k], inplace=True)

    dfenownow = (dfenow.groupby([xvar, yvar])['True_Prevalence_elim'].sum() / num_seeds).reset_index()
    matnow = dfenownow.pivot_table(index=[yvar], columns=[xvar], values='True_Prevalence_elim')

    fig = px.imshow(matnow,
                    labels=dict(x=xvar, y=yvar, color="Elim frac"),
                    x=[str(lab) for lab in sv3_vals],
                    y=[str(lab) for lab in sv4_vals]
                    )

    fig.update_layout(transition_duration=500)

    return fig

##
if __name__ == '__main__':
   app.run_server(debug=True)