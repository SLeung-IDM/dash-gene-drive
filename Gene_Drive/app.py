import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from components.about import about
from components.header import header
from components.footer import footer
from components.page_not_found import page_not_found


external_stylesheets = [dbc.themes.BOOTSTRAP,
                        'https://codepen.io/chriddyp/pen/bWLwgP.css']
external_scripts = ['https://code.jquery.com/jquery-3.2.1.slim.min.js',
                    'https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js',
                    'https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js']

# create an instant of a dash app
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts,
                suppress_callback_exceptions=True)

# A function to wrap a component with header and footer
def layout(component=None):
    return html.Div(children=[
        header,
        component,
        footer
    ])

# -- spatial, classic, GM only, EIR = 10
wi_name = 'spatialinside_classic3allele_GM_only_aEIR10_sweep_rc_d_rr0_sne_release_day_release_node_num'
wi_name_sh = 'spatial, classic drive, GM only, EIR = 10'
data_dir = 'Z:\\home\\sleung\\workitems\\d2b\\2a2\\f47\\d2b2a2f4-77f9-eb11-a9ed-b88303911bc1'

num_sweep_vars = 6  # choose 4, 6
num_partition_vars = 3  # choose 0, 1, 3
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
        allvardefs = {'rc': 1, 'd': 1, 'rr0': 0, 'sne': 0}
        allvarvals = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                      'd': [1, 0.95, 0.9],
                      'rr0': [0, 0.1, 0.2],
                      'sne': [0, 0.05, 0.1, 0.15, 0.2]}
        allvarvals_fns = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                          'd': [1, 0.95, 0.9],
                          'rr0': [0, 0.1, 0.2],
                          'sne': [0, 0.05, 0.1, 0.15, 0.2]}
    elif drive_type == 'integral':
        allvardefs = {'rc': 1, 'd1': 1, 'rr20': 0, 'se2': 0}
        allvarvals = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                      'd1': [1, 0.95, 0.9],
                      'rr20': [0, 0.1, 0.2],
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
if 'Unnamed: 0' in dfi.columns:
    dfi = dfi.drop('Unnamed: 0', axis=1)
# if 'Unnamed: 0' in dfa.columns:
#     dfa = dfa.drop('Unnamed: 0', axis=1)
if 'Unnamed: 0' in dfe.columns:
    dfe = dfe.drop('Unnamed: 0', axis=1)
if 'Unnamed: 0' in dfed.columns:
    dfed = dfed.drop('Unnamed: 0', axis=1)
# dfa.rename(columns={'Time': 'time'}, inplace=True)
dfe.rename(columns={'Time': 'time'}, inplace=True)
dfed.rename(columns={'Time': 'time'}, inplace=True)

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
    if drive_type == 'classic':
        dfi = dfi[['Time', 'rc', 'd', 'rr0', 'sne', 'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std']]
    elif drive_type == 'integral':
        dfi = dfi[['Time', 'rc', 'd1', 'rr20', 'se2', 'PfHRP2 Prevalence', 'PfHRP2 Prevalence_std']]

gene_drive_component =  html.Div([
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

# define the home_page
# replace sample_chart with your own chart or component
home_page = layout(gene_drive_component)

# define the about_page
about_page = layout(about)

# define the error page
error_page = layout(page_not_found)

# initiate the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


# add callbacks for page navigation
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return home_page
    elif pathname == '/about':
        return about_page
    else:
        return error_page

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
            # dfednow = dfednow[dfednow['True_Prevalence_elim'] == True]
            dfednow.drop(columns=['True_Prevalence_elim'], inplace=True)
            dfednownow = (dfednow.groupby([mat_xvar, mat_yvar])['True_Prevalence_elim_day'].mean()).reset_index()
            matnow = dfednownow.pivot_table(index=[mat_yvar], columns=[mat_xvar],
                                            values='True_Prevalence_elim_day', dropna=False)
            # matnow = matnow.round(1)  # .astype('Int64')
            matnow = (matnow/365).round(1)  # .astype('Int64')
            # z_text = [[str(y) for y in x] for x in matnow.values]

            # - Create annotated heatmap
            subplots.append(ff.create_annotated_heatmap(
                z=matnow.values,
                x=matnow.columns.tolist(),
                y=matnow.index.tolist(),
                # annotation_text=z_text,
                zmax=(dfed['True_Prevalence_elim_day']/365).max(),
                zmin=(dfed['True_Prevalence_elim_day']/365).min(),
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

if __name__ == '__main__':
    app.run_server(debug=True)
