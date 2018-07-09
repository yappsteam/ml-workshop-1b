#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yappsteam
"""

import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import sys
init_notebook_mode(connected=True)

np.random.seed(5)

def plot_data(data, y):
    """
    Plots random data from the data set
    
    arguments
    --------
    data -- data set
    y -- labels
    
    output
    ------
    plot
    """
    m, _ = data.shape
    plt.figure(1)
    plt.figure(figsize=(10,10))

    n_fig = 20

    for f in range(1, n_fig + 1):
        plt.subplot(5, 4, f)
        idx = np.random.randint(1, m)
        plt.imshow(np.reshape(data[idx], (8, 8)))
        plt.title('target = {}'.format(y[idx]))
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.90, hspace=0.35, wspace=0.35)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_tuning(data):
    with plt.style.context('fivethirtyeight'):
        plt.figure(figsize=(7,7))
        plt.scatter(data['gamma'].values, data['f1_train'].values, s = 100)
        plt.scatter(data['gamma'].values, data['f1_dev'].values, s = 100)
        plt.legend(['Train Set', 'Dev Set'], loc = 0)
        plt.title('Train vs Dev Set')
        plt.xlabel(r'$ gamma $')
        plt.ylabel('Accuracy')
        plt.show()

def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))

def plot_misc(model, X, y):
    """
    Create a plot using the model and data
    
    """
    # Get the shape
    m, _ = X.shape
    plt.figure(1)
    
    plt.figure(figsize=(10,10))
    
    # predictions
    y_pred = model.predict(X)
    X = X[y != y_pred]
    n_fig = X.shape[0]
    
    # slicing y's
    y_hat = y_pred[y_pred != y]
    y_ = y[y_pred != y]
    
    for f in range(1, n_fig + 1):
        plt.subplot(2, 2, f)
        idx = f - 1 #np.random.randint(1, m)
        plt.imshow(np.reshape(X[idx], (8, 8)))
        plt.title(r'y = {} | $\hat y$ = {}'.format(y_[idx], y_hat[idx]))
    plt.subplots_adjust(top=0.72, bottom=0.08, left=0.10, right=0.90, hspace=0.35, wspace=0.35)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_predictions(model, X, y):
    """
    Create a plot using the model and data
    
    """
    # Get the shape
    m, _ = X.shape
    plt.figure(1)
    
    plt.figure(figsize=(10,10))
    n_fig = 20
    
    # predictions
    y_pred = model.predict(X)
    
    for f in range(1, n_fig + 1):
        plt.subplot(5, 4, f)
        idx = np.random.randint(1, m)
        plt.imshow(np.reshape(X[idx], (8, 8)))
        plt.title(r'y = {} | $\hat y$ = {}'.format(y[idx], y_pred[idx]))
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.90, hspace=0.35, wspace=0.35)
    plt.show()
    
def iplot_tuning(plot_data):

    trace0 = go.Scatter(
            x = plot_data['gamma'].values,
            y = plot_data['f1_train'].values,
            mode = 'lines+markers',
            name = 'Train Set')
    
    trace1 = go.Scatter(
            x = plot_data['gamma'].values,
            y = plot_data['f1_dev'].values,
            mode = 'lines+markers',
            name = 'Dev Set')
    
    layout = go.Layout(
            title='Train vs Dev set',
            xaxis=dict(
                    title='gamma hyperparameter',
                    titlefont=dict(
                            family='Courier New, monospace',
                            size=18,
                            color='#7f7f7f'
                            )
                    ),
            yaxis=dict(
                    title='Accuracy',
                    titlefont=dict(
                            family='Courier New, monospace',
                            size=18,
                            color='#7f7f7f'
                            )
                    )
                    )
        
    data = [trace0, trace1]
    configure_plotly_browser_state()
    #iplot(data, filename='scatter-mode')
    iplot({"data": data, "layout": layout}, filename='scatter-mode')
    #return data
    #iplot(data, filename='scatter-mode')
