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

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__)

##
# -------- Choose experiment and set up params
svs_by_drive_type = {
    'Classic': ['rc', 'd', 'rr0', 'sne'],
    'Integral': ['rc', 'd1', 'rr20', 'se2'],
}

sv_vals_by_drive_type = {
    'Classic': {
        'rc': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'd': [0.9, 0.95, 1.0],
        'rr0': [0.0, 0.001, 0.01, 0.1],
        'sne': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
    'Integral': {
        'rc': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'd1': [0.9, 0.95, 1.0],
        'rr20': [0.0, 0.001, 0.01, 0.1],
        'se2': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
}

sv_defs_by_drive_type = {
    'Classic': {
        'rc': 1.0, 'd': 1.0, 'sne': 0.0, 'rr0': 0.0
    },
    'Integral': {
        'rc': 1.0, 'd1': 1.0, 'se2': 0.0, 'rr20': 0.0
    }
}

eirs_itns = [
    'EIR = 10, no ITNs',
    'EIR = 10, with ITNs',
    'EIR = 30, no ITNs',
    'EIR = 30, with ITNs',
    'EIR = 80, with ITNs',
]

fns_by_drive_type_eir_itn = {
    'Classic': {
        'EIR = 10, no ITNs': 'spatialinside_classic3allele_GM_only_aEIR10_sweep_rc_d_rr0_sne',
        'EIR = 10, with ITNs': 'spatialinside_classic3allele_VC_and_GM_aEIR10_sweep_rc_d_rr0_sne',
        'EIR = 30, no ITNs': 'spatialinside_classic3allele_GM_only_aEIR30_sweep_rc_d_rr0_sne',
        'EIR = 30, with ITNs': 'spatialinside_classic3allele_VC_and_GM_aEIR30_sweep_rc_d_rr0_sne',
        'EIR = 80, with ITNs': 'spatialinside_classic3allele_VC_and_GM_aEIR80_sweep_rc_d_rr0_sne'
    },
    'Integral': {
        'EIR = 10, no ITNs': 'spatialinside_integral2l4a_GM_only_aEIR10_sweep_rc_d1_rr20_se2',
        'EIR = 10, with ITNs': 'spatialinside_integral2l4a_VC_and_GM_aEIR10_sweep_rc_d1_rr20_se2',
        'EIR = 30, no ITNs': 'spatialinside_integral2l4a_GM_only_aEIR30_sweep_rc_d1_rr20_se2',
        'EIR = 30, with ITNs': 'spatialinside_integral2l4a_VC_and_GM_aEIR30_sweep_rc_d1_rr20_se2',
        'EIR = 80, with ITNs': 'spatialinside_integral2l4a_VC_and_GM_aEIR80_sweep_rc_d1_rr20_se2'
    }
}

itn_distrib_days = [180, 3 * 365 + 180, 6 * 365 + 180]
released_day = 180
num_yrs = 8  # length of sim
num_seeds = 20  # num of seeds per sim
elim_day = 2555  # day on which elim fraction is calculated

data_dir = 'csvs'

##
# -------- Load data
dfis = {}
# dfas = {}
dfes = {}
dfeds = {}
for drive_typenow in fns_by_drive_type_eir_itn.keys():
    for eir_itnnow in fns_by_drive_type_eir_itn[drive_typenow].keys():
        winame = fns_by_drive_type_eir_itn[drive_typenow][eir_itnnow]
        dfis[winame] = pd.read_csv(os.path.join(data_dir, 'dfi_' + winame + '.csv'))
        dfis[winame]['Infectious Vectors Num'] = dfis[winame]['Adult Vectors'] * dfis[winame]['Infectious Vectors']
        # dfas[winame] = pd.read_csv(os.path.join(data_dir, 'dfa_' + winame + '.csv'))
        dfes[winame] = pd.read_csv(os.path.join(data_dir, 'dfe_' + winame + '.csv'))
        dfeds[winame] = pd.read_csv(os.path.join(data_dir, 'dfed_' + winame + '.csv'))

##
# -------- Dash
app.layout = html.Div([

    dcc.Tabs([

        dcc.Tab(label='Elimination probability matrices', children=[

            html.H2(children='Elimination probabilities'),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['EIR and ITNs:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='eir-itn0',
                        options=[{'label': i, 'value': i} for i in list(eirs_itns)],
                        value='EIR = 30, with ITNs'
                    )
                ], style={'width': '20%'}),

                html.Div(children=[
                    html.Label(['Drive type:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='drive-type0',
                        options=[{'label': i, 'value': i} for i in list(svs_by_drive_type.keys())],
                        value='Integral'
                    )
                ], style={'width': '20%'})

            ], style=dict(display='flex')),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-xvar0')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-yvar0')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Matrix x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='matrix-xvar0')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Matrix y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='matrix-yvar0')
                ], style={'width': '10%'})

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='elim-prob-matrices',
                          style={'width': '95%', 'height': '80vh'})
            ])
        ]),

        dcc.Tab(label='Years to elimination matrices', children=[

            html.H2(children='Years to elimination'),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['EIR and ITNs:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='eir-itn1',
                        options=[{'label': i, 'value': i} for i in list(eirs_itns)],
                        value='EIR = 30, with ITNs'
                    )
                ], style={'width': '20%'}),

                html.Div(children=[
                    html.Label(['Drive type:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='drive-type1',
                        options=[{'label': i, 'value': i} for i in list(svs_by_drive_type.keys())],
                        value='Integral'
                    )
                ], style={'width': '20%'})

            ], style=dict(display='flex')),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-xvar1')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-yvar1')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Matrix x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='matrix-xvar1')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Matrix y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='matrix-yvar1')
                ], style={'width': '10%'}),

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='elim-time-matrices',
                          style={'width': '95%', 'height': '80vh'})
            ])
        ]),

        dcc.Tab(label='PfHRP2 prevalence time series', children=[

            html.H2(children='PfHRP2 prevalence'),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['EIR and ITNs:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='eir-itn2',
                        options=[{'label': i, 'value': i} for i in list(eirs_itns)],
                        value='EIR = 30, with ITNs'
                    )
                ], style={'width': '20%'}),

                html.Div(children=[
                    html.Label(['Drive type:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='drive-type2',
                        options=[{'label': i, 'value': i} for i in list(svs_by_drive_type.keys())],
                        value='Integral'
                    )
                ], style={'width': '20%'})

            ], style=dict(display='flex')),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-xvar2')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-yvar2')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['1st plot var (color):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var2-0')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['2nd plot var (line style):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var2-1')
                ], style={'width': '10%'}),

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='prev-ts',
                          style={'width': '100%', 'height': '80vh'})
            ])
        ]),


    ])
])


# ---------------------------------------------
# Callbacks for dropdowns to update dropdowns
# ---------------------------------------------
# ---- Elim prob matrices
@app.callback(
    [Output('outer-xvar0', 'options'),
     Output('outer-yvar0', 'options'),
     Output('matrix-xvar0', 'options'),
     Output('matrix-yvar0', 'options')],
    [Input('drive-type0', 'value')])
def set_sv_options(sel_drive_type):
    outer_xvar_opts = svs_by_drive_type[sel_drive_type]
    outer_yvar_opts = svs_by_drive_type[sel_drive_type]
    matrix_xvar_opts = svs_by_drive_type[sel_drive_type]
    matrix_yvar_opts = svs_by_drive_type[sel_drive_type]
    return [{'label': i, 'value': i} for i in outer_xvar_opts], \
           [{'label': i, 'value': i} for i in outer_yvar_opts], \
           [{'label': i, 'value': i} for i in matrix_xvar_opts], \
           [{'label': i, 'value': i} for i in matrix_yvar_opts]


@app.callback(
    [Output('outer-xvar0', 'value'),
     Output('outer-yvar0', 'value'),
     Output('matrix-xvar0', 'value'),
     Output('matrix-yvar0', 'value')],
    [Input('outer-xvar0', 'options'),
     Input('outer-yvar0', 'options'),
     Input('matrix-xvar0', 'options'),
     Input('matrix-yvar0', 'options')])
def set_sv_value(outer_xvar_opts, outer_yvar_opts, matrix_xvar_opts, matrix_yvar_opts):
    return outer_xvar_opts[0]['value'], outer_yvar_opts[1]['value'], \
           matrix_xvar_opts[2]['value'], matrix_yvar_opts[3]['value']


# ---- Elim time matrices
@app.callback(
    [Output('outer-xvar1', 'options'),
     Output('outer-yvar1', 'options'),
     Output('matrix-xvar1', 'options'),
     Output('matrix-yvar1', 'options')],
    [Input('drive-type1', 'value')])
def set_sv_options(sel_drive_type):
    outer_xvar_opts = svs_by_drive_type[sel_drive_type]
    outer_yvar_opts = svs_by_drive_type[sel_drive_type]
    matrix_xvar_opts = svs_by_drive_type[sel_drive_type]
    matrix_yvar_opts = svs_by_drive_type[sel_drive_type]
    return [{'label': i, 'value': i} for i in outer_xvar_opts], \
           [{'label': i, 'value': i} for i in outer_yvar_opts], \
           [{'label': i, 'value': i} for i in matrix_xvar_opts], \
           [{'label': i, 'value': i} for i in matrix_yvar_opts]


@app.callback(
    [Output('outer-xvar1', 'value'),
     Output('outer-yvar1', 'value'),
     Output('matrix-xvar1', 'value'),
     Output('matrix-yvar1', 'value')],
    [Input('outer-xvar1', 'options'),
     Input('outer-yvar1', 'options'),
     Input('matrix-xvar1', 'options'),
     Input('matrix-yvar1', 'options')])
def set_sv_value(outer_xvar_opts, outer_yvar_opts, matrix_xvar_opts, matrix_yvar_opts):
    return outer_xvar_opts[0]['value'], outer_yvar_opts[1]['value'], \
           matrix_xvar_opts[2]['value'], matrix_yvar_opts[3]['value']


# ---- Prev ts
@app.callback(
    [Output('outer-xvar2', 'options'),
     Output('outer-yvar2', 'options'),
     Output('sweep-var-2-0', 'options'),
     Output('sweep-var-2-1', 'options')],
    [Input('drive-type2', 'value')])
def set_sv_options(sel_drive_type):
    outer_xvar_opts = svs_by_drive_type[sel_drive_type]
    outer_yvar_opts = svs_by_drive_type[sel_drive_type]
    matrix_xvar_opts = svs_by_drive_type[sel_drive_type]
    matrix_yvar_opts = svs_by_drive_type[sel_drive_type]
    return [{'label': i, 'value': i} for i in outer_xvar_opts], \
           [{'label': i, 'value': i} for i in outer_yvar_opts], \
           [{'label': i, 'value': i} for i in matrix_xvar_opts], \
           [{'label': i, 'value': i} for i in matrix_yvar_opts]


@app.callback(
    [Output('outer-xvar2', 'value'),
     Output('outer-yvar2', 'value'),
     Output('sweep-var-2-0', 'value'),
     Output('sweep-var-2-1', 'value')],
    [Input('outer-xvar2', 'options'),
     Input('outer-yvar2', 'options'),
     Input('sweep-var-2-0', 'options'),
     Input('sweep-var-2-1', 'options')])
def set_sv_value(outer_xvar_opts, outer_yvar_opts, matrix_xvar_opts, matrix_yvar_opts):
    return outer_xvar_opts[0]['value'], outer_yvar_opts[1]['value'], \
           matrix_xvar_opts[2]['value'], matrix_yvar_opts[3]['value']


# ---------------------------------------------
# Callbacks for figures
# ---------------------------------------------
# ---- Elim prob matrices
@app.callback(
    Output('elim-prob-matrices', 'figure'),
    [Input('eir-itn0', 'value'),
     Input('drive-type0', 'value'),
     Input('outer-xvar0', 'value'),
     Input('outer-yvar0', 'value'),
     Input('matrix-xvar0', 'value'),
     Input('matrix-yvar0', 'value')])
def update_elim_prob_matrices(sel_eir_itn, sel_drive_type,
                              ov_xvar, ov_yvar, mat_xvar, mat_yvar):
    # - Get selected data and sweep var vals
    svvals = sv_vals_by_drive_type[sel_drive_type]
    svdefs = sv_defs_by_drive_type[sel_drive_type]
    winame = fns_by_drive_type_eir_itn[sel_drive_type][sel_eir_itn]
    dfe = dfes[winame]

    # - Get all outer xvar and yvar vals
    ov_xvar_vals = svvals[ov_xvar]
    ov_yvar_vals = svvals[ov_yvar]

    # - Compute subplot titles and heatmaps
    iaxis = 1
    subplots = []

    dfesm = dfe[dfe[mat_xvar].isin(svvals[mat_xvar]) &
                dfe[mat_yvar].isin(svvals[mat_yvar])]

    for ov_yvar_val in ov_yvar_vals:
        for ov_xvar_val in ov_xvar_vals:

            # - Compute heatmap values
            allvardefsnow = {k: v for k, v in svdefs.items() if k not in [mat_xvar, mat_yvar, ov_xvar, ov_yvar]}
            dfenow = dfesm
            if len(allvardefsnow) > 0:
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
                x=list(range(len(svvals[mat_xvar]))),
                y=list(range(len(svvals[mat_yvar]))),
                zmin=0,
                zmax=1,
                showscale=True,
                colorscale='YlOrBr_r')
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
        tickmode='array',
        tickvals=list(range(len(svvals[mat_xvar]))),
        ticktext=[str(val) for val in svvals[mat_xvar]]
    )
    fig.update_yaxes(
        tickmode='array',
        tickvals=list(range(len(svvals[mat_yvar]))),
        ticktext=[str(val) for val in svvals[mat_yvar]]
    )
    fig.update_layout(margin=dict(l=60, r=50, b=50, t=30))

    return fig


# ---- Elim time matrices
@app.callback(
    Output('elim-time-matrices', 'figure'),
    [Input('eir-itn1', 'value'),
     Input('drive-type1', 'value'),
     Input('outer-xvar1', 'value'),
     Input('outer-yvar1', 'value'),
     Input('matrix-xvar1', 'value'),
     Input('matrix-yvar1', 'value')])
def update_elim_time_matrices(sel_eir_itn, sel_drive_type,
                              ov_xvar, ov_yvar, mat_xvar, mat_yvar):
    # - Get selected data and sweep var vals
    svvals = sv_vals_by_drive_type[sel_drive_type]
    svdefs = sv_defs_by_drive_type[sel_drive_type]
    winame = fns_by_drive_type_eir_itn[sel_drive_type][sel_eir_itn]
    dfed = dfeds[winame]

    # - Get all outer xvar and yvar vals
    ov_xvar_vals = svvals[ov_xvar]
    ov_yvar_vals = svvals[ov_yvar]

    # - Compute subplot titles and heatmaps
    iaxis = 1
    subplots = []

    dfedsm = dfed[dfed[mat_xvar].isin(svvals[mat_xvar]) &
                  dfed[mat_yvar].isin(svvals[mat_yvar])]

    for ov_yvar_val in ov_yvar_vals:
        for ov_xvar_val in ov_xvar_vals:

            # - Compute heatmap values
            svdefsnow = {k: v for k, v in svdefs.items() if k not in [mat_xvar, mat_yvar, ov_xvar, ov_yvar]}
            dfednow = dfedsm
            if len(svdefsnow) > 0:
                for k, v in svdefsnow.items():
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
            matnow = (matnow / 365).round(1)  # .astype('Int64')

            # - Create annotated heatmap
            subplots.append(ff.create_annotated_heatmap(
                z=matnow.values,
                x=list(range(len(svvals[mat_xvar]))),
                y=list(range(len(svvals[mat_yvar]))),
                # zmin=(dfed['True_Prevalence_elim_day'] / 365).min(),
                # zmax=(dfed['True_Prevalence_elim_day'] / 365).max(),
                zmin=2.5,
                zmax=num_yrs,
                showscale=True,
                colorscale='YlOrBr')
            )

            # - Update annotation axes
            for annot in subplots[-1]['layout']['annotations']:
                annot['xref'] = 'x' + str(iaxis)
                annot['yref'] = 'y' + str(iaxis)
            iaxis = iaxis + 1

    # - Set up subplot framework
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

    # - Update annotations for all subplots
    for isp, subplot in enumerate(subplots):
        fig.layout.annotations += subplots[isp].layout.annotations

    # - Update fig layout and subplot axes
    fig.update_xaxes(
        ticklen=10,
        tickmode='array',
        tickvals=list(range(len(svvals[mat_xvar]))),
        ticktext=[str(val) for val in svvals[mat_xvar]]
    )
    fig.update_yaxes(
        ticklen=10,
        tickmode='array',
        tickvals=list(range(len(svvals[mat_yvar]))),
        ticktext=[str(val) for val in svvals[mat_yvar]]
    )
    fig.update_layout(margin=dict(l=60, r=50, b=50, t=30))

    return fig


# ---- Prevalence time series
@app.callback(
    Output('prev-ts', 'figure'),
    [Input('eir-itn2', 'value'),
     Input('drive-type2', 'value'),
     Input('outer-xvar2', 'value'),
     Input('outer-yvar2', 'value'),
     Input('sweep-var2-0', 'value'),
     Input('sweep-var2-1', 'value')])
def update_prev_ts(sel_eir_itn, sel_drive_type,
                   ov_xvar, ov_yvar, svar0, svar1):
    # - Get selected data and sweep var vals
    svvals = sv_vals_by_drive_type[sel_drive_type]
    svdefs = sv_defs_by_drive_type[sel_drive_type]
    winame = fns_by_drive_type_eir_itn[sel_drive_type][sel_eir_itn]
    dfi = dfis[winame]

    # - Subset dataframe
    dfinow = dfi[dfi[svar0].isin(svvals[svar0]) &
                 dfi[svar1].isin(svvals[svar1]) &
                 dfi[ov_xvar].isin(svvals[ov_xvar]) &
                 dfi[ov_yvar].isin(svvals[ov_yvar])]
    svdefsnow = {k: v for k, v in svdefs.items() if k not in [svar0, svar1, ov_xvar, ov_yvar]}
    for k, v in svdefsnow.items():
        dfinow = dfinow[dfinow[k] == v]
        dfinow.drop(columns=[k], inplace=True)

    # - Plot
    fig = px.line(dfinow, x='Time', y='PfHRP2 Prevalence',
                  labels={
                      'PfHRP2 Prevalence': ''
                  },
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row="all", col="all",
                  annotation_text="Vector release",
                  annotation_position="bottom right")
    if 'with ITN' in sel_eir_itn:
        for distrib_day in itn_distrib_days:
            fig.add_vline(x=distrib_day, line_dash="dot", line_color="green",
                          row="all", col="all",
                          annotation_text="ITN distrib",
                          annotation_position="bottom right")
    return fig


'''
@app.callback(
    Output('av-ts', 'figure'),
    [Input('outer-xvar3', 'value'),
     Input('outer-yvar3', 'value'),
     Input('sweep-var3-0', 'value'),
     Input('sweep-var3-1', 'value')])
def update_av_ts(ov_xvar, ov_yvar, svar0, svar1):
    dfinow = dfi[dfi[svar0].isin(allvarvals[svar0]) &
                 dfi[svar1].isin(allvarvals[svar1]) &
                 dfi[ov_xvar].isin(allvarvals[ov_xvar]) &
                 dfi[ov_yvar].isin(allvarvals[ov_yvar])]

    allvardefsnow = {k: v for k, v in allvardefs.items() if k not in [svar0, svar1, ov_xvar, ov_yvar]}
    for k, v in allvardefsnow.items():
        dfinow = dfinow[dfinow[k] == v]
        dfinow.drop(columns=[k], inplace=True)

    fig = px.line(dfinow, x='Time', y='Adult Vectors',
                  labels={
                      'Adult Vectors': '#'
                  },
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    return fig


@app.callback(
    Output('ivf-ts', 'figure'),
    [Input('outer-xvar4', 'value'),
     Input('outer-yvar4', 'value'),
     Input('sweep-var4-0', 'value'),
     Input('sweep-var4-1', 'value')])
def update_ivf_ts(ov_xvar, ov_yvar, svar0, svar1):
    dfinow = dfi[dfi[svar0].isin(allvarvals[svar0]) &
                 dfi[svar1].isin(allvarvals[svar1]) &
                 dfi[ov_xvar].isin(allvarvals[ov_xvar]) &
                 dfi[ov_yvar].isin(allvarvals[ov_yvar])]

    allvardefsnow = {k: v for k, v in allvardefs.items() if k not in [svar0, svar1, ov_xvar, ov_yvar]}
    for k, v in allvardefsnow.items():
        dfinow = dfinow[dfinow[k] == v]
        dfinow.drop(columns=[k], inplace=True)

    fig = px.line(dfinow, x='Time', y='Infectious Vectors',
                  labels={
                      'Infectious Vectors': ''
                  },
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    return fig


@app.callback(
    Output('ivn-ts', 'figure'),
    [Input('outer-xvar5', 'value'),
     Input('outer-yvar5', 'value'),
     Input('sweep-var5-0', 'value'),
     Input('sweep-var5-1', 'value')])
def update_ivn_ts(ov_xvar, ov_yvar, svar0, svar1):
    dfinow = dfi[dfi[svar0].isin(allvarvals[svar0]) &
                 dfi[svar1].isin(allvarvals[svar1]) &
                 dfi[ov_xvar].isin(allvarvals[ov_xvar]) &
                 dfi[ov_yvar].isin(allvarvals[ov_yvar])]

    allvardefsnow = {k: v for k, v in allvardefs.items() if k not in [svar0, svar1, ov_xvar, ov_yvar]}
    for k, v in allvardefsnow.items():
        dfinow = dfinow[dfinow[k] == v]
        dfinow.drop(columns=[k], inplace=True)

    fig = px.line(dfinow, x='Time', y='Infectious Vectors Num',
                  labels={
                      'Infectious Vectors Num': '#'
                  },
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    return fig


@app.callback(
    Output('ef-ts', 'figure'),
    [Input('outer-xvar6', 'value'),
     Input('outer-yvar6', 'value'),
     Input('sweep-var6-0', 'value'),
     Input('sweep-var6-1', 'value')])
def update_ef_ts(ov_xvar, ov_yvar, svar0, svar1):
    dfanow = dfa[dfa[svar0].isin(allvarvals[svar0]) &
                 dfa[svar1].isin(allvarvals[svar1]) &
                 dfa[ov_xvar].isin(allvarvals[ov_xvar]) &
                 dfa[ov_yvar].isin(allvarvals[ov_yvar])]

    allvardefsnow = {k: v for k, v in allvardefs.items() if k not in [svar0, svar1, ov_xvar, ov_yvar]}
    for k, v in allvardefsnow.items():
        dfanow = dfanow[dfanow[k] == v]
        dfanow.drop(columns=[k], inplace=True)

    fig = px.line(dfanow, x='Time', y=effector_allele,
                  labels={
                      effector_allele: '',
                  },
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    return fig
'''

##
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
