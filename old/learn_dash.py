# https://www.datacamp.com/community/tutorials/learn-build-dash-python
# pip install dash==0.21.1
# pip install dash-renderer==0.13.0
# pip install dash-html-components==0.11.0
# pip install dash-core-components==0.23.0
# pip install plotly --upgrade

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash()

app.layout = html.Div([
    dcc.Input(id='my-id', value='Dash App', type='text'),
    html.Div(id='my-div')
])


@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    return 'You\'ve entered "{}"'.format(input_value)


if __name__ == '__main__':
    app.run_server()