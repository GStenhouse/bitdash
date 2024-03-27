################################################
# Capstone Project - Bitcoin Trading Dashboard #
################################################

# This is a script of a Streamlit app visualises Bitcoin trading data and provides trading recommendations for users.

# The app uses the Alpha Vantage API and will have the following features:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
####################
# Import Libraries #
####################
import psycopg2
import psycopg2.extras as extras
from datetime import datetime, timedelta
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import streamlit as st
from streamlit_extras.stylable_container import stylable_container


###############
# page set up #
###############
st.set_page_config(page_title = "Charts", layout="wide")

#############
# Functions #
#############

# set database configuration - cached
@st.cache_data
def set_db_config():
    db_config = {
        'dbname': st.secrets['database'],
        'user': st.secrets['user'],
        'password': st.secrets['password'],
        'host': st.secrets['host']}
    return db_config


# Connect to database and get latest data from database - cached
def get_latest_date():
    with psycopg2.connect(**db_config) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT date FROM student.btc_trading_data ORDER BY date DESC LIMIT 1;")
        latest_date = cur.fetchall()
    return latest_date[0][0]

# Get data from database - cached
@st.cache_data
def get_data():
    with psycopg2.connect(**db_config) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM student.btc_trading_data;")
        data = cur.fetchall()
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = 'btc_trading_data';")
        colms = cur.fetchall()
        cols = [str(elem).replace("(", "").replace(")", "").replace(",", "").replace("'", "") for elem in colms]
        dataframe = pd.DataFrame(data, columns = cols)

    return dataframe
    

###########################
# Collect and update data #
###########################

## Check database for latest date ##
# set database configuration
from configparser import ConfigParser

db_config = set_db_config()

# connect to database and get latest data from database
latest_date = get_latest_date()

# if latest date is not today, update database with latest data
if latest_date.strftime('%Y-%m-%d') != datetime.today().strftime('%Y-%m-%d'):
    # get data from Alpha Vantage API
    print('Database not up to date')
    print('Collecting data...')
    api_key = st.secrets['alphavantage_api_key']
    # set up API connection and function
    crypt_class = CryptoCurrencies(key=api_key, output_format='pandas')
    ti_class = TechIndicators(key=api_key, output_format='pandas')
    
    # download data
    btc_data = crypt_class.get_digital_currency_daily(symbol='BTC', market='GBP')[0]
    ti_data = pd.DataFrame()  # create an empty dataframe to store the technical indicators

    # Relative strength index
    ti_data['RSI'] = ti_class.get_rsi(symbol='BTC', interval='daily', time_period=14, series_type='close')[0]

    # Simple Moving Averages
    ti_data['SMA_20d'] = ti_class.get_sma(symbol='BTC', interval='daily', time_period=20, series_type='close')[0]
    ti_data['SMA_50d'] = ti_class.get_sma(symbol='BTC', interval='daily', time_period=50, series_type='close')[0]
    ti_data['SMA_100d'] = ti_class.get_sma(symbol='BTC', interval='daily', time_period=100, series_type='close')[0]
    ti_data['SMA_200d'] = ti_class.get_sma(symbol='BTC', interval='daily', time_period=200, series_type='close')[0]

    # exponential moving averages
    ti_data['EMA_9d'] = ti_class.get_ema(symbol='BTC', interval='daily', time_period=9, series_type='close')[0]
    ti_data['EMA_12d'] = ti_class.get_ema(symbol='BTC', interval='daily', time_period=20, series_type='close')[0]
    ti_data['EMA_26d'] = ti_class.get_ema(symbol='BTC', interval='daily', time_period=50, series_type='close')[0]

    # Stochastic oscillator
    stoch = ti_class.get_stoch(symbol='BTC', interval='daily', fastkperiod=14, slowkperiod=3, slowdperiod=3)[0]
    ti_data['STOCH_SlowK'] = stoch['SlowK']
    ti_data['STOCH_SlowD'] = stoch['SlowD']

    print('Data collected')
    # rest index to have date as a column
    btc_data.reset_index(inplace=True)
    ti_data.reset_index(inplace=True)

    # remove already existing data
    btc_data = btc_data[btc_data['date'] > str(latest_date)]
    ti_data = ti_data[ti_data['date'] > str(latest_date)]

    # combine into a single dataframe, removing any unnecessary columns
    data = btc_data[['date', '1a. open (GBP)', '2a. high (GBP)', '3a. low (GBP)', '4a. close (GBP)', '5. volume', '6. market cap (USD)']].merge(ti_data, 'left', on = 'date')

    # rename columns to match database
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'market_cap_usd', 'RSI', 'SMA_20d', 'SMA_50d', 'SMA_100d', 'SMA_200d', 'EMA_9d', 'EMA_12d', 'EMA_26d', 'STOCH_SlowK', 'STOCH_SlowD']

    # upload data to database
    values = data.values.tolist()  # turn data into a list of lists (rows) for uploading
    cols = ', '.join(list(data.columns))  # get column names
    insert_query = f"INSERT INTO student.btc_trading_data({cols}) VALUES %s"  # SQL query to execute

    # connect to database and upload data
    print('Uploading data...')
    try:
        with psycopg2.connect(**db_config) as conn:
            cur = conn.cursor()
            extras.execute_values(cur, insert_query, values)
            conn.commit()
        print('Data uploaded successfully')
        del data
        del btc_data
        del ti_data
    except Exception as e:
        print('Data upload failed: ', e) 

    # update latest date
    latest_date = datetime.today().strftime('%Y-%m-%d')

else:
    pass


# Once database is definitely up to date, cache data - sort by date
curr_data = get_data()
curr_data = curr_data.sort_values('date', ascending = True)

##################
# Visualisations #
##################
# show latest update on the app
st.write("Updated: ", latest_date)

st.title('Bitcoin Trading Dashboard')

## Add disclaimer banner
with stylable_container(key = "disclaimer",
    css_styles = """
    {
        background-color: #FF7D75;
        border: 2px solid FF7D75;
        padding: 10px 10px 10px 10px
        }
        """
    ):
    st.write('This app is for educational purposes only and we do not recommend using this app to make financial decisions.')

## display latest data ##
# put chart in container
container = st.container()

## Add select slider to change date ranges of charts
date_range = st.select_slider('Select date range:\n',options = curr_data['date'], value = (curr_data['date'].min(), curr_data['date'].max()))

# set base plot
fig = go.Figure()
fig.update_xaxes(title_text='Date', range = [date_range[0] - timedelta(days=5), date_range[1] + timedelta(days=5)])
fig.update_yaxes(title_text='Price of Bitcoin in GBP', range = [curr_data['low'].min() - 2000, curr_data['high'].max() + 2000])
fig.update_layout(annotations=[], overwrite=True)
fig.update_layout(
    showlegend=True,
    legend=dict(orientation="h"),
    plot_bgcolor="white",
    autosize=False,
    width=1130,
    height=500,
    margin=dict(t=10,l=20,b=10,r=20)
)
 
# add check boxes to show different daily measures
# do inside container so can display below graph which still influencing date range of graphs
with st.container(border = True):
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        open = st.checkbox('Show open price', value = True)
    with col2: 
        close = st.checkbox('Show close price')
    with col3: 
        high = st.checkbox('Show high price')
    with col4:
        low = st.checkbox('Show low price')

    # show line based on checkbox
    lines = []
    if open:
        fig.add_trace(go.Scatter(x=curr_data['date'], y=curr_data['open'], mode='lines', name='Open', line=dict(color='dodgerblue')))
    if close:
        fig.add_trace(go.Scatter(x=curr_data['date'], y=curr_data['close'], mode='lines', name='Close', line=dict(color='midnightblue')))   
    if high:
        fig.add_trace(go.Scatter(x=curr_data['date'], y=curr_data['high'], mode='lines', name='High', line=dict(color='darkgreen')))
    if low:
        fig.add_trace(go.Scatter(x=curr_data['date'], y=curr_data['low'], mode='lines', name='Low', line=dict(color='crimson')))

container.plotly_chart(fig)


## display techinical indicators ##
st.subheader('Technical Indicators')

# put in tabs so can easily see each chart compared to the main chart
tab1, tab2, tab3, tab4 = st.tabs(["Relative Strength Index", "Simple Moving Average", "Exponential Moving Average", "Stochastic Oscilator"])

# Set up common figure parameters
params = {'figure.figsize': (6, 2),
         'axes.labelsize': 4,
         'xtick.labelsize': 4,
         'ytick.labelsize': 4,
         'lines.linewidth': 0.9,
         'xtick.major.width': 0.5,
         'ytick.major.width': 0.5}
plt.rcParams.update(params)

## RSI
# set base plot
with tab1:
    fig, ax = plt.subplots(figsize=(6, 1.5), dpi = 100)
    plt.setp(ax.spines.values(), linewidth=0.4)

    ax.set_xlim(date_range[0] - timedelta(days=5), date_range[1] + timedelta(days=5))
    ax.set_ylim(0, 100)

    # add line for RSI and 70 and 30 thresholds
    sns.lineplot(data=curr_data, x='date', y='rsi', color = 'red', linewidth = 0.6)
    sns.lineplot(x=curr_data['date'], y=70, color = 'slategray', linewidth = 0.6)
    sns.lineplot(x=curr_data['date'], y=30, color = 'slategray', linewidth = 0.6)

    ax.lines[1].set_linestyle("--")
    ax.lines[2].set_linestyle("--")

    plt.text(date_range[1] + timedelta(days=7), 67, 'Overbought', color = 'slategray', size = 5)
    plt.text(date_range[1] + timedelta(days=7), 27, 'Oversold', color = 'slategray', size = 5)

    plt.xlabel('Date', size = 6) 
    plt.ylabel('Relative strength index', size = 6) 

    st.pyplot(fig.figure, use_container_width=False)
    plt.close()

    # add container for recommendations
    rsi_container = st.container()

    # add expander for further details on relative strength index
    with st.expander(r"$\textsf{\large Further details}$"):
        st.write("Insert description of RSI here")


## Stochastic Oscillator
with tab2:
    fig, ax = plt.subplots(figsize=(6, 1.5), dpi = 100)
    plt.setp(ax.spines.values(), linewidth=0.4)

    ax.set_xlim(date_range[0] - timedelta(days=5), date_range[1] + timedelta(days=5))
    ax.set_ylim(0, 100)

    # add fast and slow stochastic oscillator lines and 70 and 30 thresholds
    palette = sns.color_palette("YlOrBr_r", 5)
    sns.lineplot(data=curr_data, x='date', y='stoch_slowk', color = palette[1], linewidth = 0.6)
    sns.lineplot(data=curr_data, x='date', y='stoch_slowd', color = palette[3], linewidth = 0.6)
    sns.lineplot(x=curr_data['date'], y=80, color = 'slategray', linewidth = 0.6)
    sns.lineplot(x=curr_data['date'], y=20, color = 'slategray', linewidth = 0.6)

    ax.lines[2].set_linestyle("--")
    ax.lines[3].set_linestyle("--")

    plt.text(date_range[1] + timedelta(days=7), 77, 'Overbought', color = 'slategray', size = 5)
    plt.text(date_range[1] + timedelta(days=7), 17, 'Oversold', color = 'slategray', size = 5)

    plt.text(date_range[1] + timedelta(days=7), 58, 'Slow', color = palette[1], size = 5)
    plt.text(date_range[1] + timedelta(days=7), 42, 'Fast', color = palette[3], size = 5)

    plt.xlabel('Date', size = 6)
    plt.ylabel('Stochastic oscillator', size = 6)

    st.pyplot(fig.figure)
    plt.close()

    # add container for recommendations
    osc_container = st.container()

    # add expander for further details on Stochastic Oscillator
    with st.expander(r"$\textsf{\large Further details}$"):
        st.write("Insert description of Stoch here")


## Simple Moving Averages
with tab3:
    fig, ax = plt.subplots(figsize=(6, 1.6), dpi = 100)
    plt.setp(ax.spines.values(), linewidth=0.4)

    ax.set_xlim(date_range[0] - timedelta(days=5), date_range[1] + timedelta(days=5))
    ax.set_ylim(curr_data['sma_20d'].min() - 0.5, curr_data['sma_20d'].max() + 0.5)

    # add line for each SMA
    palette = sns.color_palette("viridis_r", 4)
    sns.lineplot(data=curr_data, x='date', y='sma_20d', color = palette[0], linewidth = 0.6)
    sns.lineplot(data=curr_data, x='date', y='sma_50d', color = palette[1], linewidth = 0.6)
    sns.lineplot(data=curr_data, x='date', y='sma_100d', color = palette[2], linewidth = 0.6)
    sns.lineplot(data=curr_data, x='date', y='sma_200d', color = palette[3], linewidth = 0.6)

    # add text to show which line is which
    plt.text(date_range[1] + timedelta(days=7), curr_data['sma_20d'].max() - ((curr_data['sma_20d'].max() - curr_data['sma_20d'].min()) / 8), "20 day", color = palette[0], size = 5)
    plt.text(date_range[1] + timedelta(days=7), curr_data['sma_20d'].max() - 2*((curr_data['sma_20d'].max() - curr_data['sma_20d'].min()) / 8), "50d", color = palette[1], size = 5)
    plt.text(date_range[1] + timedelta(days=7), curr_data['sma_20d'].max() - 3*((curr_data['sma_20d'].max() - curr_data['sma_20d'].min()) / 8), "100d", color = palette[2], size = 5)
    plt.text(date_range[1] + timedelta(days=7), curr_data['sma_20d'].max() - 4*((curr_data['sma_20d'].max() - curr_data['sma_20d'].min()) / 8), "200d", color = palette[3], size = 5)

    plt.xlabel('Date', size = 6) 
    plt.ylabel('Simple moving average', size = 6) 

    st.pyplot(fig.figure)
    plt.close()

    # add container for recommendations
    sma_container = st.container()

    # add expander for further details on simple moving averages
    with st.expander(r"$\textsf{\large Further details}$"):
        st.write("Insert description of SMA here")


## Exponential  Moving Averages
with tab4:
    fig, ax = plt.subplots(figsize=(6, 1.6), dpi = 100)
    plt.setp(ax.spines.values(), linewidth=0.4)

    ax.set_xlim(date_range[0] - timedelta(days=5), date_range[1] + timedelta(days=5))
    ax.set_ylim(curr_data['ema_9d'].min() - 0.5, curr_data['ema_9d'].max() + 0.5)

    # add line for each SMA
    palette = sns.color_palette("cubehelix", 4)
    sns.lineplot(data=curr_data, x='date', y='ema_9d', color = palette[0], linewidth = 0.6)
    sns.lineplot(data=curr_data, x='date', y='ema_12d', color = palette[1], linewidth = 0.6)
    sns.lineplot(data=curr_data, x='date', y='ema_26d', color = palette[2], linewidth = 0.6)

    # add text to show which line is which
    plt.text(date_range[1] + timedelta(days=7), curr_data['ema_9d'].max() - ((curr_data['ema_9d'].max() - curr_data['ema_9d'].min()) / 8), "9 day", color = palette[0], size = 5)
    plt.text(date_range[1] + timedelta(days=7), curr_data['ema_9d'].max() - 2*((curr_data['ema_9d'].max() - curr_data['ema_9d'].min()) / 8), "12 day", color = palette[1], size = 5)
    plt.text(date_range[1] + timedelta(days=7), curr_data['ema_9d'].max() - 3*((curr_data['ema_9d'].max() - curr_data['ema_9d'].min()) / 8), "26 day", color = palette[2], size = 5)

    plt.xlabel('Date', size = 6) 
    plt.ylabel('Exponential moving average', size = 6) 

    st.pyplot(fig.figure)
    plt.close()

    # add container for recommendations
    ema_container = st.container()

    # add expander for further details on exponential moving averages
    with st.expander(r"$\textsf{\large Further details}$"):
        st.write("Insert description of EMA here")


## Trading Recommendations ##
## RSI based recommendations
        

## Display trading recommendations ##
## Headers
rsi_container.write(r"$\textsf{\large RSI Recommendations}$")
sma_container.write(r"$\textsf{\large SMA Recommendations}$")
ema_container.write(r"$\textsf{\large EMA Recommendations}$")
osc_container.write(r"$\textsf{\large Stochastic Oscilator Recommendations}$")
