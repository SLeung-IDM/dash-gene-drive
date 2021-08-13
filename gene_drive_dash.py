import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import os
import pandas as pd
import plotly.express as px

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

# Write out sweep variable names and values
sv1_str = 'rc'
sv2_str = 'd'
sv3_str = 'rr0'
sv4_str = 'sne'
sv5_str = 'release_day'
sv6_str = 'num_nodes'
sv1_def = 1
sv2_def = 1
sv3_def = 0
sv4_def = 0
sv5_def = 180
sv6_def = 6
sv1_vals = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
sv2_vals = [1, 0.95, 0.9]
sv3_vals = [0, 0.1, 0.2]
sv4_vals = [0, 0.05, 0.1, 0.15, 0.2]
sv5_vals = [180, 240, 300, 360, 420, 480, 545]
sv6_vals = [6, 12]

partition_vars = [sv6_str, sv2_str, sv3_str]
partition_vars_vals = [sv6_vals, sv2_vals, sv3_vals]

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
     #dfi = dfi.append(pd.read_csv(filei))
    # dfa = dfa.append(pd.read_csv(filea))
    dfe = dfe.append(pd.read_csv(filee))
    # dfed = dfed.append(pd.read_csv(fileed))

# Clean up data
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

##
app.layout = html.Div([

    html.Div([
        dcc.Dropdown(
            id='x-axis-var',
            options=[{'label': i, 'value': i} for i in [sv1_str, sv2_str, sv3_str, sv4_str, sv5_str, sv6_str]],
            value=sv3_str)
    ], style={'width': '49%', 'padding': '0px 20px 20px 20px'}),

    html.Div([
        dcc.Dropdown(
            id='y-axis-var',
            options=[{'label': i, 'value': i} for i in [sv1_str, sv2_str, sv3_str, sv4_str, sv5_str, sv6_str]],
            value=sv4_str)
    ], style={'width': '49%', 'padding': '0px 20px 20px 20px'}),

    html.Div([
        dcc.Slider(
            id='rc-slider',
            min=dfe['rc'].min(),
            max=dfe['rc'].max(),
            value=dfe['rc'].min(),
            marks={str(rc): str(rc) for rc in dfe['rc'].unique()},
            step=None)
    ], style={'width': '49%', 'padding': '0px 20px 20px 20px'}),

    html.Div([
        dcc.Slider(
            id='d-slider',
            min=dfe['d'].min(),
            max=dfe['d'].max(),
            value=dfe['d'].min(),
            marks={str(d): str(d) for d in dfe['d'].unique()},
            step=None
        )
    ], style={'width': '49%', 'padding': '0px 20px 20px 20px'}),

    dcc.Graph(id='elim-prob-matrix')
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