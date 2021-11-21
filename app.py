from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
import argparse
from pywebio import start_server
from pywebio.session import *
#!/usr/bin/python3
from pywebio.input import *
from pywebio.output import *
from pandas.core.frame import DataFrame
import yfinance as yf
import numpy as np
import talib as ta
import pandas as pd
import numpy as np
import csv 
import re
from pywebio.output import put_html
from cutecharts.charts import Line
app = Flask(__name__)
def create_chart(labels: list, values: list):
    chart = Line("Total Scores vs Date")
    chart.set_options(labels=labels, x_label="Date", y_label="Score")
    chart.add_series("Total Scores", values)
    return chart 
def create_chart2(labels: list, values: list):
    chart = Line("Close vs Date")
    chart.set_options(labels=labels, x_label="Close", y_label="Score")
    chart.add_series("Close", values)
    return chart
def create_chart4(labels: list, values: list):
    chart = Line("Close vs Date")
    chart.set_options(labels=labels, x_label="Close", y_label="Score")
    chart.add_series("Close", values)
    return chart


def get_score(df: DataFrame, indicator: str, entry_type='long'):
	indicator = indicator.upper()
 
	if indicator == 'RSI' and entry_type == 'long':
		try:
			rsiValue = df.rsi.head(1).values[0]
			if rsiValue in range(60,70):
				return 5
			elif rsiValue in range(70,80):
				return 4
			elif rsiValue >= 80:
				return 3
			elif rsiValue in range(50,60):
				return 2
			else:
				return 0
		except IndexError:
			return 0
	if indicator == 'MACD' and entry_type == 'long':
		macd = df.macd_crossover
		try:
			date = macd.iloc[list(np.where(df["macd_crossover"] == 1)[0])].index.values[0]
			date = pd.to_datetime(date)
			dates = df.index.values
			for i in range(0,len(dates)):
				if pd.to_datetime(dates[i]).date() == date:
					return 5 - i
			return 0
		except IndexError:
			return 0
	if indicator == 'EMA' and entry_type == 'long':
		try:
			date = df.ema_crossover.iloc[list(np.where(df["ema_crossover"] == 1)[0])].index.values[0]
			date = pd.to_datetime(date)
			dates = df.index.values
			for i in range(0,len(dates)):
				if pd.to_datetime(dates[i]).date() == date:
					return 5 - i
			return 0
		except IndexError:
			return 0
	if indicator == 'VOLUME' and entry_type == 'long':
		try:
			date = df.volume_buy.iloc[list(np.where(df["volume_buy"] == 1)[0])].index.values[0]
			date = pd.to_datetime(date)
			dates = df.index.values
			for i in range(0,len(dates)):
				if pd.to_datetime(dates[i]).date() == date:
					return 5 - i
			return 0
		except IndexError:
			return 0
	return None
def lr():
    s = select("Select Mode", ['Manual','CSV'])
    if s == 'Manual':
        SE = input_group("Basic info",[input("Enter Stock Symbol",name="ticker"),input('Start Date',name='start_date')])
        
        #download data from yfinance of somedays
        data = DataFrame(yf.download(SE['ticker'], start=SE['start_date'], period="max"))
        put_html("<h1>Stock Symbol: "+SE['ticker']+"</h1>")
    else:
        userfile = file_upload('Upload csv data file')
        open(userfile['filename'],'wb').write(userfile['content'])
        put_html("<h1>Uploaded file: " + userfile['filename'] + "</h1>")
        data=pd.read_csv(userfile['filename'])
        data.set_index('Date', inplace=True)
        #printcsv name
        put_html("<h1>CSV file: " + userfile['filename'] + "</h1>")
    put_html("<h1>Head of CSV file: </h1>")
    put_html(data.head().to_html())
    put_html("<h1>Tail of CSV file: </h1>")
    put_html(data.tail().to_html())

    # take span as input
    Span = input('Enter span: ', type=NUMBER)
    data = data.tail(Span+5)

    try :
        data = data.drop(['Adj Close'], axis = 1)
    except KeyError:
        pass
    #drop rows with na 
    data = data.dropna()
    data['5EMA'] = pd.Series.ewm(data['Close'], span=5).mean()

    data['26EMA'] = pd.Series.ewm(data['Close'], span=26).mean()

    data['rsi'] = ta.RSI(data['Close'].values, timeperiod=14)

    data['macd'], data['macdSignal'], data['macdHist'] = ta.MACD(data.Close.values, fastperiod=12, slowperiod=26, signalperiod=9)

    data['macd_crossover'] = np.where(((data.macd > data.macdSignal) & (data.macd.shift(1) < data.macdSignal.shift(1))), 1, 0)
    data['macd_crossunder'] = np.where(((data.macd < data.macdSignal) & (data.macd.shift(1) > data.macdSignal.shift(1))), 1, 0)
    data['ema_crossover'] = np.where(((data['5EMA'].shift(1) <= data['26EMA'].shift(1)) & (data['5EMA'] > data['26EMA'] )), 1, 0)
    data['ema_crossunder'] = np.where(((data['5EMA'].shift(1) >= data['26EMA'].shift(1)) & (data['5EMA'] < data['26EMA'] )), 1, 0)

    data['rsi_buy'] = np.where(data.rsi > 60, 1, 0)
    data['rsi_sell'] = np.where(data.rsi < 40, 1, 0)

    data['volume_buy'] = np.where((data.Volume > data.Volume.ewm(span=5).mean()) & (data.Close > data.Close.shift(1)), 1, 0)
    data['volume_sell'] = np.where((data.Volume > data.Volume.ewm(span=5).mean()) & (data.Close < data.Close.shift(1)), 1, 0)

    last_week_data = data.tail(5).sort_values(by='Date', ascending=False)
    rsiScore = get_score(last_week_data, indicator='rsi')
    macdScore = get_score(last_week_data, indicator='macd')
    emaScore = get_score(last_week_data, indicator='ema')
    volumeScore = get_score(last_week_data, indicator='volume')
    totalScore = rsiScore + macdScore + emaScore + volumeScore

    totalScoreL = [0,0,0,0,0]
    for i in range(len(data.index.values)-5):
        df = data[i:i+5]
        rsiScore = get_score(df, indicator='rsi')
        macdScore = get_score(df, indicator='macd')
        emaScore = get_score(df, indicator='ema')
        volumeScore = get_score(df, indicator='volume')
        totalScore = rsiScore + macdScore + emaScore + volumeScore
        totalScoreL.append(totalScore)

    data = data.iloc[5:,:]

    data['totalScore'] = totalScoreL[5:]
    data['totalScore last'] = data.totalScore.ewm(span=Span).mean()

    data['dates'] = data.index.values
    data['dates'] = data['dates'].astype(str)
    #get only the dates
    data['dates'] = data['dates'].str.split('-').str[2]
    print(data.head())

    totalScoreX = data.iloc[:,-2:-1]
    datesY = data["dates"]
    datesY = datesY.astype(int)

    print(data.dtypes)
    """ manual 
    Open               float64
    High               float64
    Low                float64
    Close              float64
    Volume               int64
    5EMA               float64
    26EMA              float64
    rsi                float64
    macd               float64
    macdSignal         float64
    macdHist           float64
    macd_crossover       int64
    macd_crossunder      int64
    ema_crossover        int64
    ema_crossunder       int64
    rsi_buy              int64
    rsi_sell             int64
    volume_buy           int64
    volume_sell          int64
    totalScore           int64
    totalScore last    float64
    dates               object"""
    """
    csv
    Open               float64
    High               float64
    Low                float64
    Close              float64
    Volume             float64
    5EMA               float64
    26EMA              float64
    rsi                float64
    macd               float64
    macdSignal         float64
    macdHist           float64
    macd_crossover       int64
    macd_crossunder      int64
    ema_crossover        int64
    ema_crossunder       int64
    rsi_buy              int64
    rsi_sell             int64
    volume_buy           int64
    volume_sell          int64
    totalScore           int64
    totalScore last    float64
    dates               object
    """
    #put text len(totalScoreX),len(datesY)
    data['dates'] = data['dates'].astype(str)
    data.index = data.index.astype(str)
    put_html("<h1>Span = " + str(Span) + "</h1>")
    chart = create_chart(list(data.index),list(data.totalScore))
    put_html(chart.render_notebook())
    chart2 = create_chart2(list(data.Close),list(data.totalScore))
    put_html(chart2.render_notebook())
    chart4 = create_chart4(list(data.Close),list(data["totalScore last"]))
    put_html(chart4.render_notebook())
    put_html("<h1>Data with Span = " + str(Span) + "</h1>")
    put_html(data.to_html())
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(np.array(datesY).reshape(-1, 1), totalScoreX)

    #take date as input from user int
    inp = input('Enter date for Score prediction: ', type=NUMBER)

    yhat = model.predict([[inp]]) #predict score for next day
    put_html("<h1>R : " + str(model.score(np.array(datesY).reshape(-1, 1), totalScoreX)) + "</h1>")
    from sklearn.ensemble import RandomForestRegressor

    # create regressor object
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

    # fit the regressor with x and y data
    # x =Open	High	Low	Close	Volume
    # y = 5EMA	26EMA	rsi	macd	macdSignal	macdHist
    regressor.fit(np.asarray(data['totalScore last']).reshape(-1, 1),data['Close'])

    regressor.predict(yhat) #predict close
    put_html("<h1>R : " + str(regressor.score(np.array(datesY).reshape(-1, 1), totalScoreX)) + "</h1>")
    put_html('<h1>Predicted Score: {} on Date {} </h1> '.format(yhat, inp))
    put_html('<h1>Predicted Close: {} on Date {} </h1>'.format(regressor.predict(yhat), inp))

    #add column dates
    data['dates'] = data.Date.values

    # implement desison tree
    from sklearn.tree import DecisionTreeRegressor
    #fill na with mean of that column
    data = data.fillna(data.mean())

    model = DecisionTreeRegressor(random_state = 0)
    model.fit(np.asarray(data['totalScore last']).reshape(-1, 1),data['Close'])
    print(model.score(np.asarray(data['totalScore last']).reshape(-1, 1),data['Close']))
    model.predict(yhat) #predict close

app = Flask(__name__)

app.add_url_rule('/tool', 'webio_view', webio_view(lr),methods=['GET', 'POST', 'OPTIONS'])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()
    start_server(lr,port=args.port)