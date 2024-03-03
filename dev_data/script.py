from datetime import datetime,timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA,ARIMAResults
from tensorflow.keras.models import load_model

def get_price(name,from_date):
  df = pd.read_csv(os.path.join(os.path.dirname(__file__), f'../data/price/{name}.csv'))
  df = df.filter(['Date','Close'])
  if(from_date):
    filtered_df = df.loc[(df['Date'] >= from_date)]
  else:
    filtered_df = df
  data = filtered_df.to_json(orient='records')
  return data

def get_prediction(name,model,steps=100):
  df = pd.read_csv(f'predictions/{model}/{name}.csv')
  df = df.filter(['Date','Predicted Price'])
  date = datetime.now().strftime('%Y-%m-%d')
  filtered_df = df.loc[(df['Date'] > date)]

  if(steps<len(filtered_df)):
    filtered_df = filtered_df[:steps]

  data = filtered_df.to_json(orient='records')
  return data

def pred_multistep(x,y,steps,model,scaler):
  x_pred=x[-1:, :, :]
  y_pred=y[-1]
  n_future = steps
  y_future = []

  for i in range(n_future):
    # feed the last forecast back to the model as an input
    x_pred = np.append(x_pred[:, :, 1:], y_pred.reshape(1, 1, 1))
    x_pred = x_pred.reshape(1,1,x_pred.shape[0])

    # generate the next forecast
    y_pred = model.predict(x_pred)

    # save the forecast
    y_future.append(y_pred.flatten()[0])

  # transform the forecasts back to the original scale
  y_future = np.array(y_future).reshape(-1, 1)
  y_future = scaler.inverse_transform(y_future)
  return y_future

def load_LSTM(pair,x,y):
  path = f'../data/models/LSTM/{pair}.h5'
  model = load_model(path)
  model.fit(x,y)
  model.save(path)
  return model

def load_GRU(pair,x,y):
  path = f'../data/models/GRU/{pair}.h5'
  model = load_model(path)
  model.fit(x,y)
  model.save(path)
  return model

def predict_ARIMA(data,order,steps):
  model = ARIMA(data.filter(['Close']), order=order).fit()
  pred = model.forecast(steps)
  return pred

def reshape_timeseries(data):
  return np.reshape(data, (data.shape[0], 1, data.shape[1]))

def to_df_with_date(pred):
  date=datetime.now().date()
  dates = []
  for i in range(len(pred)):
    date = date+timedelta(days=1)
    dates.append(date)
  df = pd.DataFrame({"Date": dates})
  df["Predicted Price"]= pred
  return df

def update_prediction(x,y,df,name,steps=100):
  print('Updating new predictions...')

  pair_names=['USD_VND','EUR_VND','GBP_VND','VND_JPY']
  models = ['ARIMA','GRU','LSTM']
  arima_order = {
    'USD_VND': (10,1,10),
    'EUR_VND': (2,1,7),
    'GBP_VND': (1,0,4),
    'VND_JPY': (0,1,8)
  }
  scalers={}

  pred={}
  scalers[name]=joblib.load(f'../data/scalers/{name}.pkl')

  pred['ARIMA']=predict_ARIMA(df,arima_order[name],steps)
  pred['GRU']=pred_multistep(x,y,steps,load_GRU(name,x,y),scalers[name])
  pred['LSTM']=pred_multistep(x,y,steps,load_LSTM(name,x,y),scalers[name])
  for model in models:
    if(model=='ARIMA'):
      pred[model]=np.array(pred[model])
    pred[model]=to_df_with_date(pred[model])
    pred[model].to_csv(f'predictions/{model}/{name}.csv')
  

def create_updated_dataset(df, name, days, look_back=15):
  scaler = joblib.load(f'../data/scalers/{name}.pkl')
  data = df[-look_back-days:].filter(['Close'])
  scaled_data = scaler.transform(data)

  X, Y = [], []
  for i in range(days):
    a = scaled_data[i:look_back+i, 0]
    X.append(a)
    Y.append(scaled_data[look_back+i, 0])
  
  X = np.array(X)
  Y = np.array(Y)
  X = reshape_timeseries(X)
  return [X,Y]


def init_data():
  end = datetime.now()
  start = datetime(end.year - 7, end.month, end.day)
  symbols=['VND=X','EURVND=X','GBPVND=X','VNDJPY=X']
  pair_names=['USD_VND','EUR_VND','GBP_VND','VND_JPY']
  
  for name,symbol in zip(pair_names,symbols):
    df = yf.download(symbol, start, end)
    df.to_csv(f'data/price/{name}.csv')

def update_data():
  print('Updating new data...')
  symbols=['VND=X','EURVND=X','GBPVND=X','VNDJPY=X']
  pair_names=['USD_VND','EUR_VND','GBP_VND','VND_JPY']
  
  for name,symbol in zip(pair_names,symbols):
    old_df = pd.read_csv(f'data/price/{name}.csv',usecols=['Date'])
    end_date = datetime.now().date()
    start_date = datetime.strptime(old_df.iloc[-1]['Date'], '%Y-%m-%d')
    start_date = start_date + timedelta(1)

    if(start_date.date()==end_date):
      print('Data is up to date')
      continue

    df = yf.download(symbol, start_date, end_date)
    if((start_date-timedelta(1)).date()==df.index[-1].date()):
      print('Data is up to date')
      continue

    print(f'{len(df)} rows to update in {name}.csv')
    df.to_csv(f'data/price/{name}.csv',mode='a',header=False)
    # data = pd.read_csv(os.path.join(os.path.dirname(__file__), f'../data/price/{name}.csv'))

    data = pd.read_csv(f'data/price/{name}.csv')
    x,y = create_updated_dataset(data, name, len(df))

    update_prediction(x,y,data,name)

