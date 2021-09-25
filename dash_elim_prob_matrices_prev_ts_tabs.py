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

alleles_by_drive_type = {
    'Classic': {
        'wt_allele': 'a0',
        'effector_allele': 'a1',
        'resistance_allele': 'a2'
    },
    'Integral': {
        'wt_allele': 'b0',
        'effector_allele': 'b1',
        'resistance_allele': 'b2'
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
dfas = {}
dfes = {}
dfeds = {}
for drive_typenow in fns_by_drive_type_eir_itn.keys():
    for eir_itnnow in fns_by_drive_type_eir_itn[drive_typenow].keys():
        winame = fns_by_drive_type_eir_itn[drive_typenow][eir_itnnow]
        dfis[winame] = pd.read_csv(os.path.join(data_dir, 'dfi_' + winame + '.csv'))
        dfis[winame]['Infectious Vectors Num'] = dfis[winame]['Adult Vectors'] * dfis[winame]['Infectious Vectors']
        dfas[winame] = pd.read_csv(os.path.join(data_dir, 'dfa_' + winame + '.csv'))
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
                    html.Label(['2nd plot var (style):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var2-1')
                ], style={'width': '10%'}),

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='prev-ts',
                          style={'width': '100%', 'height': '80vh'})
            ])
        ]),

        dcc.Tab(label='Adult vectors time series', children=[

            html.H2(children='Adult vectors'),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['EIR and ITNs:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='eir-itn3',
                        options=[{'label': i, 'value': i} for i in list(eirs_itns)],
                        value='EIR = 30, with ITNs'
                    )
                ], style={'width': '20%'}),

                html.Div(children=[
                    html.Label(['Drive type:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='drive-type3',
                        options=[{'label': i, 'value': i} for i in list(svs_by_drive_type.keys())],
                        value='Integral'
                    )
                ], style={'width': '20%'})

            ], style=dict(display='flex')),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-xvar3')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-yvar3')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['1st plot var (color):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var3-0')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['2nd plot var (style):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var3-1')
                ], style={'width': '10%'}),

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='av-ts',
                          style={'width': '100%', 'height': '80vh'})
            ])
        ]),

        dcc.Tab(label='Infectious fraction time series', children=[

            html.H2(children='Infectious vector fraction'),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['EIR and ITNs:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='eir-itn4',
                        options=[{'label': i, 'value': i} for i in list(eirs_itns)],
                        value='EIR = 30, with ITNs'
                    )
                ], style={'width': '20%'}),

                html.Div(children=[
                    html.Label(['Drive type:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='drive-type4',
                        options=[{'label': i, 'value': i} for i in list(svs_by_drive_type.keys())],
                        value='Integral'
                    )
                ], style={'width': '20%'})

            ], style=dict(display='flex')),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-xvar4')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-yvar4')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['1st plot var (color):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var4-0')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['2nd plot var (style):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var4-1')
                ], style={'width': '10%'}),

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='ivf-ts',
                          style={'width': '100%', 'height': '80vh'})
            ])
        ]),

        dcc.Tab(label='Infectious vectors time series', children=[

            html.H2(children='Infectious vectors'),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['EIR and ITNs:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='eir-itn5',
                        options=[{'label': i, 'value': i} for i in list(eirs_itns)],
                        value='EIR = 30, with ITNs'
                    )
                ], style={'width': '20%'}),

                html.Div(children=[
                    html.Label(['Drive type:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='drive-type5',
                        options=[{'label': i, 'value': i} for i in list(svs_by_drive_type.keys())],
                        value='Integral'
                    )
                ], style={'width': '20%'})

            ], style=dict(display='flex')),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-xvar5')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-yvar5')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['1st plot var (color):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var5-0')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['2nd plot var (style):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var5-1')
                ], style={'width': '10%'}),

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='ivn-ts',
                          style={'width': '100%', 'height': '80vh'})
            ])
        ]),

        dcc.Tab(label='Effector frequency time series', children=[

            html.H2(children='Effector frequency (or drive+effector in classic drive case)'),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['EIR and ITNs:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='eir-itn6',
                        options=[{'label': i, 'value': i} for i in list(eirs_itns)],
                        value='EIR = 30, with ITNs'
                    )
                ], style={'width': '20%'}),

                html.Div(children=[
                    html.Label(['Drive type:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='drive-type6',
                        options=[{'label': i, 'value': i} for i in list(svs_by_drive_type.keys())],
                        value='Integral'
                    )
                ], style={'width': '20%'})

            ], style=dict(display='flex')),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-xvar6')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-yvar6')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['1st plot var (color):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var6-0')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['2nd plot var (style):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var6-1')
                ], style={'width': '10%'}),

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='ef-ts',
                          style={'width': '100%', 'height': '80vh'})
            ])
        ]),

        dcc.Tab(label='Wild type frequency time series', children=[

            html.H2(children='Wild type frequency (at effector locus in integral drive case)'),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['EIR and ITNs:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='eir-itn7',
                        options=[{'label': i, 'value': i} for i in list(eirs_itns)],
                        value='EIR = 30, with ITNs'
                    )
                ], style={'width': '20%'}),

                html.Div(children=[
                    html.Label(['Drive type:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='drive-type7',
                        options=[{'label': i, 'value': i} for i in list(svs_by_drive_type.keys())],
                        value='Integral'
                    )
                ], style={'width': '20%'})

            ], style=dict(display='flex')),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-xvar7')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-yvar7')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['1st plot var (color):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var7-0')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['2nd plot var (style):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var7-1')
                ], style={'width': '10%'}),

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='wt-ts',
                          style={'width': '100%', 'height': '80vh'})
            ])
        ]),

        dcc.Tab(label='Resistance frequency time series', children=[

            html.H2(children='Resistance frequency (at effector locus in integral drive case)'),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['EIR and ITNs:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='eir-itn8',
                        options=[{'label': i, 'value': i} for i in list(eirs_itns)],
                        value='EIR = 30, with ITNs'
                    )
                ], style={'width': '20%'}),

                html.Div(children=[
                    html.Label(['Drive type:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(
                        id='drive-type8',
                        options=[{'label': i, 'value': i} for i in list(svs_by_drive_type.keys())],
                        value='Integral'
                    )
                ], style={'width': '20%'})

            ], style=dict(display='flex')),

            html.Div(children=[

                html.Div(children=[
                    html.Label(['Outer x-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-xvar8')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['Outer y-var:'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='outer-yvar8')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['1st plot var (color):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var8-0')
                ], style={'width': '10%'}),

                html.Div(children=[
                    html.Label(['2nd plot var (style):'], style={'font-weight': 'bold', 'text-align': 'center'}),
                    dcc.Dropdown(id='sweep-var8-1')
                ], style={'width': '10%'}),

            ], style=dict(display='flex')),

            html.Div([
                dcc.Graph(id='rs-ts',
                          style={'width': '100%', 'height': '80vh'})
            ])
        ])

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
     Output('sweep-var2-0', 'options'),
     Output('sweep-var2-1', 'options')],
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
     Output('sweep-var2-0', 'value'),
     Output('sweep-var2-1', 'value')],
    [Input('outer-xvar2', 'options'),
     Input('outer-yvar2', 'options'),
     Input('sweep-var2-0', 'options'),
     Input('sweep-var2-1', 'options')])
def set_sv_value(outer_xvar_opts, outer_yvar_opts, matrix_xvar_opts, matrix_yvar_opts):
    return outer_xvar_opts[0]['value'], outer_yvar_opts[2]['value'], \
           matrix_xvar_opts[3]['value'], matrix_yvar_opts[1]['value']


# ---- Adult vector ts
@app.callback(
    [Output('outer-xvar3', 'options'),
     Output('outer-yvar3', 'options'),
     Output('sweep-var3-0', 'options'),
     Output('sweep-var3-1', 'options')],
    [Input('drive-type3', 'value')])
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
    [Output('outer-xvar3', 'value'),
     Output('outer-yvar3', 'value'),
     Output('sweep-var3-0', 'value'),
     Output('sweep-var3-1', 'value')],
    [Input('outer-xvar3', 'options'),
     Input('outer-yvar3', 'options'),
     Input('sweep-var3-0', 'options'),
     Input('sweep-var3-1', 'options')])
def set_sv_value(outer_xvar_opts, outer_yvar_opts, matrix_xvar_opts, matrix_yvar_opts):
    return outer_xvar_opts[0]['value'], outer_yvar_opts[2]['value'], \
           matrix_xvar_opts[3]['value'], matrix_yvar_opts[1]['value']


# ---- Infectious vector fraction ts
@app.callback(
    [Output('outer-xvar4', 'options'),
     Output('outer-yvar4', 'options'),
     Output('sweep-var4-0', 'options'),
     Output('sweep-var4-1', 'options')],
    [Input('drive-type4', 'value')])
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
    [Output('outer-xvar4', 'value'),
     Output('outer-yvar4', 'value'),
     Output('sweep-var4-0', 'value'),
     Output('sweep-var4-1', 'value')],
    [Input('outer-xvar4', 'options'),
     Input('outer-yvar4', 'options'),
     Input('sweep-var4-0', 'options'),
     Input('sweep-var4-1', 'options')])
def set_sv_value(outer_xvar_opts, outer_yvar_opts, matrix_xvar_opts, matrix_yvar_opts):
    return outer_xvar_opts[0]['value'], outer_yvar_opts[2]['value'], \
           matrix_xvar_opts[3]['value'], matrix_yvar_opts[1]['value']


# ---- Infectious vectors ts
@app.callback(
    [Output('outer-xvar5', 'options'),
     Output('outer-yvar5', 'options'),
     Output('sweep-var5-0', 'options'),
     Output('sweep-var5-1', 'options')],
    [Input('drive-type5', 'value')])
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
    [Output('outer-xvar5', 'value'),
     Output('outer-yvar5', 'value'),
     Output('sweep-var5-0', 'value'),
     Output('sweep-var5-1', 'value')],
    [Input('outer-xvar5', 'options'),
     Input('outer-yvar5', 'options'),
     Input('sweep-var5-0', 'options'),
     Input('sweep-var5-1', 'options')])
def set_sv_value(outer_xvar_opts, outer_yvar_opts, matrix_xvar_opts, matrix_yvar_opts):
    return outer_xvar_opts[0]['value'], outer_yvar_opts[2]['value'], \
           matrix_xvar_opts[3]['value'], matrix_yvar_opts[1]['value']


# ---- Effector freq ts
@app.callback(
    [Output('outer-xvar6', 'options'),
     Output('outer-yvar6', 'options'),
     Output('sweep-var6-0', 'options'),
     Output('sweep-var6-1', 'options')],
    [Input('drive-type6', 'value')])
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
    [Output('outer-xvar6', 'value'),
     Output('outer-yvar6', 'value'),
     Output('sweep-var6-0', 'value'),
     Output('sweep-var6-1', 'value')],
    [Input('outer-xvar6', 'options'),
     Input('outer-yvar6', 'options'),
     Input('sweep-var6-0', 'options'),
     Input('sweep-var6-1', 'options')])
def set_sv_value(outer_xvar_opts, outer_yvar_opts, matrix_xvar_opts, matrix_yvar_opts):
    return outer_xvar_opts[0]['value'], outer_yvar_opts[2]['value'], \
           matrix_xvar_opts[3]['value'], matrix_yvar_opts[1]['value']


# ---- Wild type freq ts
@app.callback(
    [Output('outer-xvar7', 'options'),
     Output('outer-yvar7', 'options'),
     Output('sweep-var7-0', 'options'),
     Output('sweep-var7-1', 'options')],
    [Input('drive-type7', 'value')])
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
    [Output('outer-xvar7', 'value'),
     Output('outer-yvar7', 'value'),
     Output('sweep-var7-0', 'value'),
     Output('sweep-var7-1', 'value')],
    [Input('outer-xvar7', 'options'),
     Input('outer-yvar7', 'options'),
     Input('sweep-var7-0', 'options'),
     Input('sweep-var7-1', 'options')])
def set_sv_value(outer_xvar_opts, outer_yvar_opts, matrix_xvar_opts, matrix_yvar_opts):
    return outer_xvar_opts[0]['value'], outer_yvar_opts[2]['value'], \
           matrix_xvar_opts[3]['value'], matrix_yvar_opts[1]['value']


# ---- Resistance freq ts
@app.callback(
    [Output('outer-xvar8', 'options'),
     Output('outer-yvar8', 'options'),
     Output('sweep-var8-0', 'options'),
     Output('sweep-var8-1', 'options')],
    [Input('drive-type8', 'value')])
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
    [Output('outer-xvar8', 'value'),
     Output('outer-yvar8', 'value'),
     Output('sweep-var8-0', 'value'),
     Output('sweep-var8-1', 'value')],
    [Input('outer-xvar8', 'options'),
     Input('outer-yvar8', 'options'),
     Input('sweep-var8-0', 'options'),
     Input('sweep-var8-1', 'options')])
def set_sv_value(outer_xvar_opts, outer_yvar_opts, matrix_xvar_opts, matrix_yvar_opts):
    return outer_xvar_opts[0]['value'], outer_yvar_opts[2]['value'], \
           matrix_xvar_opts[3]['value'], matrix_yvar_opts[1]['value']


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
                      'PfHRP2 Prevalence': '',
                      'Time': 'Day',
                  },
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row="all", col="all")
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row=0, col=len(ov_xvar) - 1, annotation_text="Vector <br>release",
                  annotation_position="top right", annotation_font_color="black")
    if 'with ITN' in sel_eir_itn:
        fig.add_vline(x=itn_distrib_days[1], line_dash="dot", line_color="forestgreen",
                      row=0, col=len(ov_xvar) - 1, annotation_text="ITN <br>distribs",
                      annotation_position="top right", annotation_font_color="forestgreen")
        for distrib_day in itn_distrib_days:
            fig.add_vline(x=distrib_day, line_dash="dot", line_color="forestgreen",
                          row="all", col="all")
    fig.update_xaxes(range=[0, num_yrs * 365])
    fig.update_yaxes(range=[-0.05, 0.7])
    return fig


# ---- Adult vector time series
@app.callback(
    Output('av-ts', 'figure'),
    [Input('eir-itn3', 'value'),
     Input('drive-type3', 'value'),
     Input('outer-xvar3', 'value'),
     Input('outer-yvar3', 'value'),
     Input('sweep-var3-0', 'value'),
     Input('sweep-var3-1', 'value')])
def update_av_ts(sel_eir_itn, sel_drive_type,
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
    fig = px.line(dfinow, x='Time', y='Adult Vectors',
                  labels={
                      'Adult Vectors': '#',
                      'Time': 'Day',
                  },
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row="all", col="all")
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row=0, col=len(ov_xvar) - 1, annotation_text="Vector <br>release",
                  annotation_position="top right", annotation_font_color="black")
    if 'with ITN' in sel_eir_itn:
        fig.add_vline(x=itn_distrib_days[1], line_dash="dot", line_color="forestgreen",
                      row=0, col=len(ov_xvar) - 1, annotation_text="ITN <br>distribs",
                      annotation_position="top right", annotation_font_color="forestgreen")
        for distrib_day in itn_distrib_days:
            fig.add_vline(x=distrib_day, line_dash="dot", line_color="forestgreen",
                          row="all", col="all")
    fig.update_xaxes(range=[0, num_yrs * 365])
    fig.update_yaxes(range=[-50, 4000])
    return fig


# ---- Infectious vector fraction time series
@app.callback(
    Output('ivf-ts', 'figure'),
    [Input('eir-itn4', 'value'),
     Input('drive-type4', 'value'),
     Input('outer-xvar4', 'value'),
     Input('outer-yvar4', 'value'),
     Input('sweep-var4-0', 'value'),
     Input('sweep-var4-1', 'value')])
def update_ivf_ts(sel_eir_itn, sel_drive_type,
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
    fig = px.line(dfinow, x='Time', y='Infectious Vectors',
                  labels={
                      'Infectious Vectors': '',
                      'Time': 'Day',
                  },
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row="all", col="all")
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row=0, col=len(ov_xvar) - 1, annotation_text="Vector <br>release",
                  annotation_position="top right", annotation_font_color="black")
    if 'with ITN' in sel_eir_itn:
        fig.add_vline(x=itn_distrib_days[1], line_dash="dot", line_color="forestgreen",
                      row=0, col=len(ov_xvar) - 1, annotation_text="ITN <br>distribs",
                      annotation_position="top right", annotation_font_color="forestgreen")
        for distrib_day in itn_distrib_days:
            fig.add_vline(x=distrib_day, line_dash="dot", line_color="forestgreen",
                          row="all", col="all")
    fig.update_xaxes(range=[0, num_yrs * 365])
    fig.update_yaxes(range=[-0.01, 0.12])
    return fig


# ---- Infectious vector numbers time series
@app.callback(
    Output('ivn-ts', 'figure'),
    [Input('eir-itn5', 'value'),
     Input('drive-type5', 'value'),
     Input('outer-xvar5', 'value'),
     Input('outer-yvar5', 'value'),
     Input('sweep-var5-0', 'value'),
     Input('sweep-var5-1', 'value')])
def update_ivn_ts(sel_eir_itn, sel_drive_type,
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
    fig = px.line(dfinow, x='Time', y='Infectious Vectors Num',
                  labels={
                      'Infectious Vectors Num': '#',
                      'Time': 'Day',
                  },
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row="all", col="all")
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row=0, col=len(ov_xvar) - 1, annotation_text="Vector <br>release",
                  annotation_position="top right", annotation_font_color="black")
    if 'with ITN' in sel_eir_itn:
        fig.add_vline(x=itn_distrib_days[1], line_dash="dot", line_color="forestgreen",
                      row=0, col=len(ov_xvar) - 1, annotation_text="ITN <br>distribs",
                      annotation_position="top right", annotation_font_color="forestgreen")
        for distrib_day in itn_distrib_days:
            fig.add_vline(x=distrib_day, line_dash="dot", line_color="forestgreen",
                          row="all", col="all")
    fig.update_xaxes(range=[0, num_yrs * 365])
    fig.update_yaxes(range=[-5, 165])
    return fig


# ---- Effector freq time series
@app.callback(
    Output('ef-ts', 'figure'),
    [Input('eir-itn6', 'value'),
     Input('drive-type6', 'value'),
     Input('outer-xvar6', 'value'),
     Input('outer-yvar6', 'value'),
     Input('sweep-var6-0', 'value'),
     Input('sweep-var6-1', 'value')])
def update_ef_ts(sel_eir_itn, sel_drive_type,
                 ov_xvar, ov_yvar, svar0, svar1):
    # - Get selected data and sweep var vals
    svvals = sv_vals_by_drive_type[sel_drive_type]
    svdefs = sv_defs_by_drive_type[sel_drive_type]
    winame = fns_by_drive_type_eir_itn[sel_drive_type][sel_eir_itn]
    effallele = alleles_by_drive_type[sel_drive_type]['effector_allele']
    dfa = dfas[winame]

    # - Subset dataframe
    dfanow = dfa[dfa[svar0].isin(svvals[svar0]) &
                 dfa[svar1].isin(svvals[svar1]) &
                 dfa[ov_xvar].isin(svvals[ov_xvar]) &
                 dfa[ov_yvar].isin(svvals[ov_yvar])]
    svdefsnow = {k: v for k, v in svdefs.items() if k not in [svar0, svar1, ov_xvar, ov_yvar]}
    for k, v in svdefsnow.items():
        dfanow = dfanow[dfanow[k] == v]
        dfanow.drop(columns=[k], inplace=True)

    # - Plot
    fig = px.line(dfanow, x='Time', y=effallele,
                  labels={
                      effallele: '',
                      'Time': 'Day',
                  },
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row="all", col="all")
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row=0, col=len(ov_xvar) - 1, annotation_text="Vector <br>release",
                  annotation_position="top right", annotation_font_color="black")
    if 'with ITN' in sel_eir_itn:
        fig.add_vline(x=itn_distrib_days[1], line_dash="dot", line_color="forestgreen",
                      row=0, col=len(ov_xvar) - 1, annotation_text="ITN <br>distribs",
                      annotation_position="top right", annotation_font_color="forestgreen")
        for distrib_day in itn_distrib_days:
            fig.add_vline(x=distrib_day, line_dash="dot", line_color="forestgreen",
                          row="all", col="all")
    fig.update_xaxes(range=[0, num_yrs * 365])
    fig.update_yaxes(range=[-0.06, 1.06])
    return fig


# ---- Wild type freq time series
@app.callback(
    Output('wt-ts', 'figure'),
    [Input('eir-itn7', 'value'),
     Input('drive-type7', 'value'),
     Input('outer-xvar7', 'value'),
     Input('outer-yvar7', 'value'),
     Input('sweep-var7-0', 'value'),
     Input('sweep-var7-1', 'value')])
def update_wt_ts(sel_eir_itn, sel_drive_type,
                 ov_xvar, ov_yvar, svar0, svar1):
    # - Get selected data and sweep var vals
    svvals = sv_vals_by_drive_type[sel_drive_type]
    svdefs = sv_defs_by_drive_type[sel_drive_type]
    winame = fns_by_drive_type_eir_itn[sel_drive_type][sel_eir_itn]
    wtallele = alleles_by_drive_type[sel_drive_type]['wt_allele']
    dfa = dfas[winame]

    # - Subset dataframe
    dfanow = dfa[dfa[svar0].isin(svvals[svar0]) &
                 dfa[svar1].isin(svvals[svar1]) &
                 dfa[ov_xvar].isin(svvals[ov_xvar]) &
                 dfa[ov_yvar].isin(svvals[ov_yvar])]
    svdefsnow = {k: v for k, v in svdefs.items() if k not in [svar0, svar1, ov_xvar, ov_yvar]}
    for k, v in svdefsnow.items():
        dfanow = dfanow[dfanow[k] == v]
        dfanow.drop(columns=[k], inplace=True)

    # - Plot
    fig = px.line(dfanow, x='Time', y=wtallele,
                  labels={
                      wtallele: '',
                      'Time': 'Day',
                  },
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row="all", col="all")
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row=0, col=len(ov_xvar) - 1, annotation_text="Vector <br>release",
                  annotation_position="top right", annotation_font_color="black")
    if 'with ITN' in sel_eir_itn:
        fig.add_vline(x=itn_distrib_days[1], line_dash="dot", line_color="forestgreen",
                      row=0, col=len(ov_xvar) - 1, annotation_text="ITN <br>distribs",
                      annotation_position="top right", annotation_font_color="forestgreen")
        for distrib_day in itn_distrib_days:
            fig.add_vline(x=distrib_day, line_dash="dot", line_color="forestgreen",
                          row="all", col="all")
    fig.update_xaxes(range=[0, num_yrs * 365])
    fig.update_yaxes(range=[-0.06, 1.06])
    return fig


# ---- Resistance freq time series
@app.callback(
    Output('rs-ts', 'figure'),
    [Input('eir-itn8', 'value'),
     Input('drive-type8', 'value'),
     Input('outer-xvar8', 'value'),
     Input('outer-yvar8', 'value'),
     Input('sweep-var8-0', 'value'),
     Input('sweep-var8-1', 'value')])
def update_rs_ts(sel_eir_itn, sel_drive_type,
                 ov_xvar, ov_yvar, svar0, svar1):
    # - Get selected data and sweep var vals
    svvals = sv_vals_by_drive_type[sel_drive_type]
    svdefs = sv_defs_by_drive_type[sel_drive_type]
    winame = fns_by_drive_type_eir_itn[sel_drive_type][sel_eir_itn]
    rsallele = alleles_by_drive_type[sel_drive_type]['rs_allele']
    dfa = dfas[winame]

    # - Subset dataframe
    dfanow = dfa[dfa[svar0].isin(svvals[svar0]) &
                 dfa[svar1].isin(svvals[svar1]) &
                 dfa[ov_xvar].isin(svvals[ov_xvar]) &
                 dfa[ov_yvar].isin(svvals[ov_yvar])]
    svdefsnow = {k: v for k, v in svdefs.items() if k not in [svar0, svar1, ov_xvar, ov_yvar]}
    for k, v in svdefsnow.items():
        dfanow = dfanow[dfanow[k] == v]
        dfanow.drop(columns=[k], inplace=True)

    # - Plot
    fig = px.line(dfanow, x='Time', y=rsallele,
                  labels={
                      rsallele: '',
                      'Time': 'Day',
                  },
                  color=svar0, line_dash=svar1,
                  facet_col=ov_xvar, facet_row=ov_yvar)
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row="all", col="all")
    fig.add_vline(x=released_day, line_dash="dash", line_color="black",
                  row=0, col=len(ov_xvar) - 1, annotation_text="Vector <br>release",
                  annotation_position="top right", annotation_font_color="black")
    if 'with ITN' in sel_eir_itn:
        fig.add_vline(x=itn_distrib_days[1], line_dash="dot", line_color="forestgreen",
                      row=0, col=len(ov_xvar) - 1, annotation_text="ITN <br>distribs",
                      annotation_position="top right", annotation_font_color="forestgreen")
        for distrib_day in itn_distrib_days:
            fig.add_vline(x=distrib_day, line_dash="dot", line_color="forestgreen",
                          row="all", col="all")
    fig.update_xaxes(range=[0, num_yrs * 365])
    fig.update_yaxes(range=[-0.06, 1.06])
    return fig


##
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
