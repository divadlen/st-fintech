import streamlit as st 
from streamlit import session_state as state 

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import yfinance as yf
import numpy as np
import pandas as pd

from datetime import datetime, date, timedelta

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def main():
    state['ticker'] = state.get('ticker', '^GSPC')
    state['start_time'] = state.get('start_time', date.today())
    state['end_time'] = state.get('end_time', date.today())
    state['rolling'] = state.get('rolling', 40)
    state['df'] = state.get('df', None)
    state['df_replace'] = state.get('df_replace', None)

    state['silhouette_plot'] = state.get('silhouette_plot', None)
    state['regime_plot'] = state.get('regime_plot', None) 
    state['regime_labels'] = state.get('regime_labels', None)
    state['fr_fig'] = state.get('fr_fig', None)
    state['br_fig'] = state.get('br_fig', None)

    st.title('Market')
    get_yf_form()


    if state['df'] is not None:
        with st.expander('Silhouette Scores'):
            show_silhoutte_form()


    if state['df'] is not None:
        with st.expander('Regime'):
            show_regime_form()

        if state['regime_labels'] is not None:
            show_regime_returns_form()
       





def get_yf_form():
    TICKER_OPTIONS = [
        '^GSPC',
        'AAPL',
        'AMZN',
        'GOOG',
        'TSLA',
        'FB',
        'MSFT',
        'NFLX',
        'NVDA',
        'SQ',
        'TWTR',
        'TSM',
        'UNH',
        'WMT',
    ]

    with st.form('yf_form'):
        ticker = st.selectbox('Select ticker', options=TICKER_OPTIONS)
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            start_time = st.date_input('Start time', value=date.today()-timedelta(days=360))
        with c2:
            end_time = st.date_input('End time', value=date.today()-timedelta(days=30))
        with c3:
            rolling = st.slider('Rolling window', min_value=1, max_value=63, value=21)

        if st.form_submit_button('Get stock data'):
            start= datetime.strftime(start_time, '%Y-%m-%d')
            end = datetime.strftime(end_time, '%Y-%m-%d')
            df, df_replace = get_spx(ticker=ticker, start_time=start, end_time=end, rolling=rolling)            
            state['ticker'] = ticker
            state['start_time'] = start_time
            state['end_time'] = end_time
            state['rolling'] = rolling
            state['df'] = df
            state['df_replace'] = df_replace




def show_silhoutte_form():
    df = state['df']
    if st.button('Calculate optimal number of clusters'):
        kmeans_silhouette = calculate_silhouette(df, model_type='kmeans', max_clusters=10)
        gmm_silhouette = calculate_silhouette(df, model_type='gmm', max_clusters=10)
        fig = plot_silhouette(kmeans_silhouette, gmm_silhouette, title='Silhouette Scores')
        state['silhouette_plot'] = fig
    
    if state['silhouette_plot'] is not None:
        fig = state['silhouette_plot']
        st.plotly_chart(fig, use_container_width=True)
         


def show_regime_form():
    df = state['df']
    df_replace = state['df_replace']

    with st.form(key=f'regime_form'):
        c1, c2 = st.columns([1,1])
        with c1:
            select_model = st.selectbox('Select model', ['KMeans', 'GMM'], key=f'regime_model')
        with c2:
            clusters = st.slider('Select number of clusters', 2, 8, key=f'regime_clusters')

        if st.form_submit_button(label=f'Submit'):
            if select_model == 'KMeans':
                model = fit_kmeans(df, clusters)
            else:
                model = fit_gmm(df, clusters)

            labels = model.predict(df)
            colors = px.colors.qualitative.Plotly
            regime_plot = create_regime_plot(df_replace, labels, colors)
            state['regime_labels'] = labels
            state['regime_plot'] = regime_plot
        
        if state['regime_plot'] is not None:
            regime_plot = state['regime_plot']
            st.plotly_chart(regime_plot, use_container_width=True)

    
def show_regime_returns_form():
    df = state['df']
    df_replace = state['df_replace']
    labels = state['regime_labels']
    interval_dict = {
        '1 Day': 1,
        '2 Day': 2,
        '1 Week': 5,
        '1 Month': 21,
        '3 Months': 63,
        '6 Months': 126,
        '1 Year': 252,
        '2 Year': 504,
        '5 Year': 1260
    }

    with st.form(key=f'regime_returns_form'):
        col1, col2 = st.columns(2)
        with col1:
            f_interval = st.selectbox('Forward returns by period', options=interval_dict.keys(), index=4)
        with col2:
            b_interval = st.selectbox('Backward returns by period', options=interval_dict.keys(), index=4)

        if st.form_submit_button(label=f'Submit'):
            f_rolling = interval_dict[f_interval]
            b_rolling = interval_dict[b_interval]

            df_ret = df_replace.copy()
            df_ret['regime'] = labels
            df_ret[f'forward returns {f_rolling} days'] = generate_returns(df_ret['Close'], period=f_rolling)
            df_ret[f'backward returns {b_rolling} days'] = generate_returns(df_ret['Close'], period=b_rolling, forward=False)
            
            fr_fig = make_hist_chart(df_ret, xdata=f'forward returns {f_rolling} days', cdata='regime')
            br_fig = make_hist_chart(df_ret, xdata=f'backward returns {b_rolling} days', cdata='regime')
            state['fr_fig'] = fr_fig
            state['br_fig'] = br_fig

        if state['fr_fig'] is not None:
            fr_fig = state['fr_fig']
            st.plotly_chart(fr_fig, use_container_width=True)

        if state['br_fig'] is not None:
            br_fig = state['br_fig']
            st.plotly_chart(br_fig, use_container_width=True)
    































#-------
# load data
#---------

def get_spx(ticker:str='SPY', start_time:str='2022-01-01', end_time:str='2023-01-01', rolling:int=10):
    df = yf.download(ticker, start = start_time, end=end_time)
    df_replace= yf.download(ticker, start = start_time, end= end_time)
    VIX = yf.download('^VIX', start= start_time, end=end_time)

    df['daily_return'] = df['Close'] / df['Close'].shift(1) - 1
    df['relative_volume'] = df['Volume'] / df['Volume'].rolling(rolling).mean()
    df['vix'] = VIX['Close']

    df = df[ ['daily_return', 'vix', 'relative_volume'] ]
    df = df[rolling:]
    df_replace = df_replace[ ['Close'] ]
    df_replace= df_replace[rolling:]
    df_replace['vix'] = df['vix']
    df_replace['relative_volume'] = df['relative_volume']
    return [df, df_replace]



#---
# silhoute scores
#---
def fit_kmeans(df, clusters):
    model = KMeans(clusters)
    model.fit(df)
    return model


def fit_gmm(df, clusters):
    model = GaussianMixture(n_components=clusters)
    model.fit(df)
    return model


def calculate_silhouette(df, model_type, max_clusters):
    silhouette_avg = []

    for n_clusters in range(2, max_clusters + 1):
        if model_type == 'kmeans':
            model = fit_kmeans(df, n_clusters)
        elif model_type == 'gmm':
            model = fit_gmm(df, n_clusters)

        cluster_labels = model.predict(df) if model_type == 'gmm' else model.labels_
        silhouette_avg.append(silhouette_score(df, cluster_labels))

    return silhouette_avg


def plot_silhouette(silhouette_avg_kmeans, silhouette_avg_gmm, title='Silhouette Scores'):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("KMeans", "GMM"))

    fig.add_trace(
        go.Scatter(x=np.arange(2, len(silhouette_avg_kmeans) + 2), y=silhouette_avg_kmeans, name='KMeans'),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=np.arange(2, len(silhouette_avg_gmm) + 2), y=silhouette_avg_gmm, name='GMM'),
        row=1, col=2,
    )
    fig.update_layout(title=title, height=400, template='plotly_dark')
    return fig




#-----
# regime
#-----

def create_regime_plot(df:pd.DataFrame, labels:list, colors:list):
    """
    df: pd.DataFrame
        df for closing prices
    
    labels: list
        labeled regime

    """
    df['regime'] = labels
    EXCLUDE_N = 1
    row_num = len(df.columns) - EXCLUDE_N
    row_heights = [0.4] + [0.6/(row_num-1)] * (row_num-1)

    fig = make_subplots(rows=row_num, cols=1, row_heights=row_heights)
    for k in range(max(labels) + 1):
        d = df[df['regime'] == k]
        fig.add_trace(
            go.Scatter(x=d.index, y=d.iloc[:, 0].values, legendgroup=k, marker_color=colors[k], name=f'Regime {k+1}', mode='markers'),
            row=1, col=1,
        )
        for i in range(len(d.columns) - EXCLUDE_N - 1):
            fig.add_trace(
                go.Scatter(x=d.index, y=d.iloc[:, i+1].values, legendgroup=k, showlegend=False, marker_color=colors[k], name=f'Regime {k+1}', mode='markers'),
                row=i + 2, col=1
            )

    height = 600 + 200 * (len(d.columns) - EXCLUDE_N - 1)
    fig.update_layout(height=height)
    for i in range(len(d.columns) - 1):
        fig.update_yaxes(title_text=f"{d.columns[i]}", row=i + 1, col=1)
    return fig

















































#---------------
# Helper
# --------------
def make_hist_chart(df:pd.DataFrame, xdata:str, cdata:str):
  df = df.sort_values(by=[cdata]) # to order legend

  fig = px.histogram(
    df,
    x= xdata,
    color= cdata,
    hover_data = [df.index],
    marginal="rug",
    barmode= 'overlay',
    opacity= 0.4,
    histnorm= 'probability', 
    color_discrete_sequence=px.colors.qualitative.Plotly,
  )
  fig.update_layout(
      title_text= f'<b>{xdata.upper()}</b>',
      title_x=0.5,
      height=400,
  )
  fig.update_yaxes()
  fig.update_xaxes(
      title_text= '',
  )
  fig.add_vline(x= 0, line_dash= 'dot', line_color='white')
  return fig


def generate_returns(df, period=1, forward= True):
  if forward == True:
    returns_df = df.pct_change(periods= period).shift(periods= -period).mul(100).round(decimals=2)
  elif forward == False:
    returns_df = df.pct_change(periods= period).shift(periods= 0).mul(100).round(decimals=2)
  return returns_df

#-----------------------
# Env variables  
#-------------------------
legend=dict(
    x=0,
    y=1,
    title_text='',
    font=dict(
      family="Times New Roman",
      size=12,
      color="white"
    ),
    bgcolor= 'rgba(0,75,154,0.3)',
    bordercolor="ivory",
    borderwidth=1
)
  