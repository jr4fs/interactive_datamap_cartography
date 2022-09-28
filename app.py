import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import math 
import collections
#from pycocotools.coco import COCO
import pickle
import plotly.express as px
#from jupyter_dash import JupyterDash
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from os import listdir
from os.path import isfile, join
import base64
import itertools
#from whitenoise import WhiteNoise   #for serving static files on Heroku
from plotly.subplots import make_subplots


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')



def load_datamap_stats(base_path):
    '''
    Returns datamap stats recorded during training split by epochs

            Parameters:
                    base_path (str): Path to model metadata

            Returns:
                    df (DataFrame): Pandas dataframe with datamap stats: 
                                    confidence, variability and correctness
    '''
    # load logged stats from training 
    with open(base_path + 'datamaps_stats.json') as fp:
        datamap_stats_raw = json.load(fp)

    result = collections.defaultdict(list)
    for stat in datamap_stats_raw:
        result[stat['Epoch']].append(stat)
    datamap_stats = list(result.values())
    #datamap_stats = list(datamap_stats.values())

    return datamap_stats

def create_figures(base_path, title, hue_metric='correct.'):
    dataframe = pd.read_pickle(base_path+"datamap_metrics.pkl")

    dataframe['correct.'] = dataframe['correctness']
    main_metric = 'variability'
    other_metric = 'confidence'

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist()) 
    style = hue_metric if num_hues < 8 else None

    fig = go.Figure(data=[
        go.Scatter(
            x=dataframe[main_metric],
            y=dataframe[other_metric],
            mode="markers",
            marker=dict(
                colorscale='viridis',
                color=dataframe[hue],
                #size=df["MW"],
                colorbar={"title": "Correctness"},
                line={"color": "#444"},
                reversescale=True,
                sizeref=45,
                sizemode="diameter",
                opacity=0.8,
            ),
        )
    ])

    # turn off native plotly.js hover effects - make sure to use
    # hoverinfo="none" rather than "skip" which also halts events.
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_layout(
        xaxis=dict(title=main_metric),
        yaxis=dict(title=other_metric),
        plot_bgcolor='rgba(255,255,255,0.1)',
        title = title
    )

    fig2 = make_subplots(rows=1, cols=3, subplot_titles=('Confidence', 'Variability', 'Correctness'))
    fig2.add_trace(go.Histogram(histfunc="count", x=dataframe["confidence"]), row=1, col=1)
    fig2.add_trace(go.Histogram(histfunc="count", x=dataframe["variability"]), row=1, col=2)
    fig2.add_trace(go.Histogram(histfunc="count", x=dataframe["correct."]), row=1, col=3)


    fig2.update_layout(yaxis_title='Density', showlegend=False)

    return fig, fig2, dataframe

base_path ='assets/vqa_models/vqa_lxr955_animals_fromScratch_20epochs_breeds/'
title='VQA LXMERT955-Animals From Scratch - 20 epochs'
fig, fig2, dataframe = create_figures(base_path, title)

app = Dash(__name__)
server = app.server 
# server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/') 

app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
    dcc.Graph(id="graph2", figure=fig2),
    html.Iframe(id="embedded-pdf", src=base_path+"easy_question_distribution.pdf", height="400", width="600"),
    html.H1(
        children="ML Dashboard",
        style={
            "color": "black",
            "text-align": "center"
        },
    )
])


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = dataframe.iloc[num]
    #img_src = df_row['IMG_URL']
    #name = df_row['NAME']
    variability = df_row['variability']
    confidence = df_row['confidence']
    correctness = df_row['correctness']
    predictions = df_row['Predictions']
    question = df_row['Question']
    target = df_row['Target']
    img_src = df_row['Image URL']

    children = [
        html.Div(children=[
            html.Img(src=img_src, style={"width": "100%"}),
            #html.H2(f"{name}", style={"color": "darkblue"}),
            #html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={"width": "100%"}),
            html.P(f"Question: {question}"),
            html.P(f"Target: {target}"),
            html.P(f"Predictions: {predictions}"),
            html.P(f"Variability: {variability}"),
            html.P(f"Confidence: {confidence}"),
            html.P(f"Correctness: {correctness}"),
        ],
        style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children

if __name__ == "__main__":app.run_server(debug=False, host='0.0.0.0', port=8090)