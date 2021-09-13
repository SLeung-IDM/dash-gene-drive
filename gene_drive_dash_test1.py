import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

wi_name = 'spatialinside_classic3allele_VC_and_GM_aEIR30_sweep_rc_d_rr0_sne'
wi_name_sh = 'spatial, classic drive, VC and GM, EIR = 30'
distrib_itns = True
num_sweep_vars = 4
drive_type = 'classic'

data_dir = 'csvs'

if distrib_itns == True:
    itn_distrib_days = [180, 3 * 365 + 180, 6 * 365 + 180]

released_mosqs = True
if released_mosqs == True:
    released_day = 180

num_yrs = 8  # length of sim
elim_day = 2555  # day on which elim fraction is calculated
num_seeds = 20  # num of seeds per sim

if num_sweep_vars == 6:
    if drive_type == 'classic':
        allvardefs = {'rc': 1, 'd': 1, 'rr0': 0, 'sne': 0,
                      'rd': 180, 'nn': 6}
        allvarvals = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                      'd': [1, 0.95, 0.9],
                      'rr0': [0, 0.001, 0.01, 0.1],
                      'sne': [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
                      # 'rd': [180, 240, 300, 360, 420, 480, 545],
                      'rd': [180],
                      'nn': [6]}
        # 'nn': [6, 12]}
elif num_sweep_vars == 4:
    if drive_type == 'classic':
        allvardefs = {'rc': 1, 'd': 1, 'sne': 0, 'rr0': 0}
        allvarvals = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                      'd': [1, 0.95, 0.9],
                      'rr0': [0, 0.001, 0.01, 0.1],
                      'sne': [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]}
    elif drive_type == 'integral':
        allvardefs = {'rc': 1, 'd1': 1, 'se2': 0, 'rr20': 0}
        allvarvals = {'rc': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                      'd1': [1, 0.95, 0.9],
                      'rr20': [0, 0.001, 0.01, 0.1],
                      'se2': [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]}

##
# -------- Load data
dfi = pd.read_csv(os.path.join(data_dir, 'dfi_' + wi_name + '.csv'))
dfa = pd.read_csv(os.path.join(data_dir, 'dfa_' + wi_name + '.csv'))
dfe = pd.read_csv(os.path.join(data_dir, 'dfe_' + wi_name + '.csv'))
dfed = pd.read_csv(os.path.join(data_dir, 'dfed_' + wi_name + '.csv'))

##
# - Compute subplot titles and heatmaps
mat_xvar = 'rr0'
mat_yvar = 'sne'
ov_xvar = 'rc'
ov_yvar = 'd'
ov_xvar_vals = [1, 0.9, 0.8, 0.7, 0.6, 0.5]  # subset or all vals
ov_yvar_vals = [1, 0.95, 0.9]  # subset or all vals

iaxis = 1
subplots = []

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

        z = matnow.values
        # z_text = np.array(["%.2f" % x for x in z.reshape(z.size)])
        # z_text = z_text.reshape(z.shape)

        # - Create annotated heatmap
        subplots.append(ff.create_annotated_heatmap(
            z=z,
            # x=[str(i) for i in matnow.columns.tolist()],
            # y=[str(i) for i in matnow.index.tolist()],
            # x=matnow.columns.tolist(),
            # y=matnow.index.tolist(),
            x=list(range(len(allvarvals[mat_xvar]))),
            y=list(range(len(allvarvals[mat_yvar]))),
            # annotation_text=z_text,
            zmax=1,
            zmin=0,
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
    tickvals=list(range(len(allvarvals[mat_xvar]))),
    ticktext=[str(val) for val in allvarvals[mat_xvar]]
)
fig.update_yaxes(
    tickmode='array',
    tickvals=list(range(len(allvarvals[mat_yvar]))),
    ticktext=[str(val) for val in allvarvals[mat_yvar]]
)
fig.update_layout(margin=dict(l=60, r=50, b=50, t=30))

fig.show()

##
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
        matnow = dfednownow.pivot_table(index=[mat_yvar], columns=[mat_xvar], values='True_Prevalence_elim_day', dropna=False)
        matnow = matnow.round(1)  #.astype('Int64')
        # z_text = [[str(y) for y in x] for x in matnow.values]

        # - Create annotated heatmap
        subplots.append(ff.create_annotated_heatmap(
            z=matnow.values,
            x=matnow.columns.tolist(),
            y=matnow.index.tolist(),
            # annotation_text=z_text,
            zmax=dfed['True_Prevalence_elim_day'].max(),
            zmin=dfed['True_Prevalence_elim_day'].min(),
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

fig.show()
