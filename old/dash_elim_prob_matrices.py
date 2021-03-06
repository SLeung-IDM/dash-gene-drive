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
wi_name_sh = 'spatial, classic drive, GM only, EIR = 30'
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
              'sne': 0, 'rd': 180, 'nn': 6}
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
dfesm.rename(columns={'release_day': 'rd', 'num_nodes': 'nn'}, inplace=True)

##
app.layout = html.Div([

    html.H1(children='Elim probabilities: ' + wi_name_sh),

    html.Div(children=[

        html.Div(children=[
            html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
            dcc.Dropdown(
                id='outer-xvar',
                options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                value='rr0'
            )
        ], style={'width': '10%'}),

        html.Div(children=[
            html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
            dcc.Dropdown(
                id='outer-yvar',
                options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                value='sne'
            )
        ], style={'width': '10%'}),

        html.Div(children=[
            html.Label(['Matrix x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
            dcc.Dropdown(
                id='matrix-xvar',
                options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                value='rc')
        ], style={'width': '10%'}),

        html.Div(children=[
            html.Label(['Matrix y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
            dcc.Dropdown(
                id='matrix-yvar',
                options=[{'label': i, 'value': i} for i in list(allvarvals.keys())],
                value='d')
        ], style={'width': '10%'}),

    ], style=dict(display='flex')),

    html.Div([

        dcc.Graph(id='elim-prob-matrices',
                  style={'width': '95%', 'height': '80vh'})
        # style = {'width': '90vh', 'height': '90vh'})

    ])  # , style={'width': '100%', 'padding': '0px 20px 20px 20px'})
])


@app.callback(
    Output('elim-prob-matrices', 'figure'),
    [Input('outer-xvar', 'value'),
     Input('outer-yvar', 'value'),
     Input('matrix-xvar', 'value'),
     Input('matrix-yvar', 'value')])
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
            dfenow = dfesm
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
        subplot_titles=subplot_titles,
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


##
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
