<!-- HOW TO RUN -->

1. Install pipenv at current directory
pip install --user pipenv
2. Download dependencies
pipenv install
3. Enable virtual environment
pipenv shell
4. Host web server (Might take a few minutes to update new data first)
python app.py 

<!-- API routes -->
Get prices route:
/prices?pair={pair_name}&from={date}

pair_name: EUR_VND, USD_VND, GBP_VND, VND_JPY
from (Optional): Ex: 2024-01-20

Example:
Query USD_VND price data from 20/01/2024
/prices?pair=USD_VND&from=2024-01-20

Query all USD_VND price data (7 years)
/prices?pair=EUR_VND
---------------------
Get predictions route:
/predictions?pair={pair_name}&model={model}&steps={steps}

pair_name: EUR_VND, USD_VND, GBP_VND, VND_JPY
model: ARIMA, GRU, LSTM
steps: Prediction of n steps / days ahead (max 100 days)

Example:
Query USD_VND prediction price with ARIMA 10 days from today
/predictions?pair=USD_VND&model=ARIMA&steps=10

Query EUR_VND prediction price with GRU 100 days from today
/predictions?pair=EUR_VND&model=GRU&steps=100
