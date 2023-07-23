# Forecasting_Currencies
Forecasting Emerging Currencies Exchange Rates Applying Machine Learning Techniques
## Abstract
This project aims to forecast the exchange rates of ten emerging currencies (USD/BRL, USD/MXN, USD/COP, USD/ARS, USD/NGN, USD/PHP, USD/TRY, USD/RUB, USD/INR, and USD/CNY) by applying machine learning techniques like Long Short-Term Memory (LSTM), Extreme Gradient Boosting XGB Regressor, and the Pycaret library. Through the experimentation of these models, it results that each currency should have its prediction model, instead of generalizing one single model to predict all currencies.
## 1. Data Exploration
Source: https://finance.yahoo.com/currencies
start_date = '2020-01-01'
end_date = '2023-07-01'
 Retrieve financial market data of currency symbol

Currencies=['USDBRL', 'USDMXN', 'USDCOP', 'USDARS', 'USDNGN', 'USDPHP', 'USDTRY', 'USDRUB', 'USDINR', 'USDCNY']
currencies_names = ['Brazilian Real - USDBRL', 'Mexican Peso - USDMXN', 'Colombian Peso - USDCOP', 'Argentine Peso - USDARS', 'Nigerian Naira - USDNGN',
                  'Philippine Peso - USDPHP', 'Turkish Lira - USDTRY', 'Russian Ruble - USDRUB', 'Indian Rupee - USDINR', 'Chinese Yuan - USDCNY']


dataframes = []

for i in Currencies:
    data_X = yf.download(i+'=X', start=start_date, end=end_date)
    dataframes.append(data_X)#

