import numpy as np
import os
import pandas as pd
import plotly.express as px

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

## - TEST 3

# https://github.com/plotly/plotly.py/issues/2313
# fig1 = ff.create_annotated_heatmap(z)
# fig2 = ff.create_annotated_heatmap(z)
# for annot in fig2['layout']['annotations']:
#     annot['xref'] = 'x2'
# fig = make_subplots(rows=1, cols=2)
# fig.add_trace(fig1.data[0], row=1, col=1)
# fig.add_trace(fig2.data[0], row=1, col=2)
# fig.update_layout(fig1.layout)
# fig.layout.annotations += fig2.layout.annotations

from plotly.subplots import make_subplots
import plotly.figure_factory as ff

mat_xvar = 'rc'
mat_yvar = 'd'
ov_xvar = 'rr0'
ov_yvar = 'sne'
ov_xvar_vals = [0, 0.1, 0.2]  # subset or all vals
ov_yvar_vals = [0, 0.05, 0.1, 0.15]  # subset or all vals

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
            coloraxis='coloraxis',
            hovertemplate=mat_xvar + ': %{x}<br>' + mat_yvar + ': %{y}<br>Elim prob: %{z}<extra></extra>')
        )

        # - Update annotation axes
        for annot in subplots[-1]['layout']['annotations']:
            annot['xref'] = 'x' + str(iaxis)
            annot['yref'] = 'y' + str(iaxis)
        iaxis = iaxis + 1

        # - Create subplot titles
        subplot_titles.append(ov_xvar + '=' + str(ov_xvar_val) + ', ' + ov_yvar + '=' + str(ov_yvar_val))

# - Set up subplot framework and titles
fig = make_subplots(
    rows=len(ov_yvar_vals), cols=len(ov_xvar_vals),
    subplot_titles=subplot_titles,
    horizontal_spacing=0.075,
    vertical_spacing=0.075
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
fig.update_layout(coloraxis={'colorscale': 'YlOrBr_r'}, title='Elim probabilities, ' + wi_name)
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

fig.show()

## - TEST 2
# https://plotly.com/python/subplots/

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}


mat_xvar = 'rc'
mat_yvar = 'd'
ov_xvar = 'rr0'
ov_yvar = 'sne'
ov_xvar_vals = [0, 0.1, 0.2]  # subset or all vals
ov_yvar_vals = [0, 0.05, 0.1, 0.15]  # subset or all vals

subplot_titles = []
for ov_yvar_val in ov_yvar_vals:
    for ov_xvar_val in ov_xvar_vals:
        subplot_titles.append(ov_xvar + '=' + str(ov_xvar_val) + ', ' + ov_yvar + '=' + str(ov_yvar_val))

fig = make_subplots(
    rows=len(ov_yvar_vals), cols=len(ov_xvar_vals),
    subplot_titles=subplot_titles,
    horizontal_spacing=0.025,
    vertical_spacing=0.075
)

isp = 1
for irow, ov_yvar_val in enumerate(ov_yvar_vals):
    for icol, ov_xvar_val in enumerate(ov_xvar_vals):
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

        fig.add_trace(go.Heatmap(z=matnow.values,
                                 x=[str(lab) for lab in allvarvals[mat_xvar]],
                                 y=[str(lab) for lab in allvarvals[mat_yvar]],
                                 # labels=dict(x=mat_xvar, y=mat_yvar, color="Elim frac"),
                                 colorscale='viridis',
                                 coloraxis='coloraxis',
                                 hovertemplate=mat_xvar + ': %{x}<br>' + mat_yvar + ': %{y}<br>Elim prob: %{z}<extra></extra>'),
                      row=irow + 1, col=icol + 1)

        # fig.add_trace(go.Scatter(x=[str(lab) for lab in allvarvals[mat_xvar]],
        #                          y=[str(lab) for lab in allvarvals[mat_yvar]],
        #                          text=matnow.values.astype(str), mode="text"),
        #               row=irow+1, col=icol+1)

        # annotations = go.Annotations()
        # for n, row in enumerate(matnow.values):
        #     for m, val in enumerate(row):
        #         annotations.append(go.layout.Annotation(text=str(matnow.values[n][m]),
        #                                                 x=allvarvals[mat_xvar][m],
        #                                                 y=allvarvals[mat_yvar][n],
        #                                                 xref='x'+str(isp), yref='y1'+str(isp), showarrow=False))
        # isp = isp+1

        # fig.update_layout(annotations=annotations,
        #                 yaxis={'title': mat_yvar}, xaxis={'title': mat_xvar})
        # yaxis_nticks=len(allvarvals[mat_yvar]), xaxis_nticks=len(allvarvals[mat_xvar]))
        # fig.add_trace(px.imshow(matnow,
        #                         labels=dict(x=mat_xvar, y=mat_yvar, color="Elim frac"),
        #                         x=[str(lab) for lab in allvarvals[mat_xvar]],
        #                         y=[str(lab) for lab in allvarvals[mat_yvar]]),
        #               row=irow+1, col=icol+1)

fig.update_layout(coloraxis={'colorscale': 'viridis'}, title='Elim prob')
# fig.update_layout(height=500, width=700,
#                   title_text="Multiple Subplots with Titles")

fig.show()

## - TEST 1

"""
xvar = 'rr0'
yvar = 'sne'
selected_rc = 1
selected_d = 1

allvars = {sv1_str: selected_rc, sv2_str: selected_d, sv3_str: sv3_def, sv4_str: sv4_def, sv5_str: sv5_def,
           sv6_str: sv6_def}
allvars = {k: v for k, v in allvars.items() if k not in [xvar, yvar]}
dfenow = dfe.drop(columns=['Daily_EIR_elim', 'New_Clinical_Cases_elim', 'Run_Number'])
for k, v in allvars.items():
    dfenow = dfenow[dfenow[k] == v]
    dfenow.drop(columns=[k], inplace=True)

dfenownow = (dfenow.groupby([xvar, yvar])['True_Prevalence_elim'].sum() / num_seeds).reset_index()

# https://towardsdatascience.com/reshape-pandas-dataframe-with-pivot-table-in-python-tutorial-and-visualization-2248c2012a31
test = dfenownow.pivot_table(index=[yvar], columns=[xvar], values='True_Prevalence_elim')

# https://plotly.com/python/heatmaps/
fig = px.imshow(test,
                labels=dict(x=xvar, y=yvar, color="Elim frac"),
                x=[str(lab) for lab in sv3_vals],
                y=[str(lab) for lab in sv4_vals]
                )
l_0, l_1, width= 8, 5, 400
fig.update_layout(xaxis=dict(scaleanchor='y', constrain='domain'),
                  width=width + 10,  # add 50 for colorbar
                  height=int(width * l_0 / l_1)
                 )
fig.show()
"""
