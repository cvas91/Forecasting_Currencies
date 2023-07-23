# Forecasting_Currencies
Forecasting Emerging Currencies Exchange Rates Applying Machine Learning Techniques
## Abstract
This project aims to forecast the exchange rates of ten emerging currencies (USD/BRL, USD/MXN, USD/COP, USD/ARS, USD/NGN, USD/PHP, USD/TRY, USD/RUB, USD/INR, and USD/CNY) by applying machine learning techniques like Long Short-Term Memory (LSTM), Extreme Gradient Boosting XGB Regressor, and the Pycaret library. Through the experimentation of these models, it results that each currency should have its prediction model, instead of generalizing one single model to predict all currencies.
## 1. Data Exploration

```python
# Source: https://finance.yahoo.com/currencies
start_date = '2020-01-01'
end_date = '2023-07-01'
# Retrieve financial market data of currency symbol

Currencies=['USDBRL', 'USDMXN', 'USDCOP', 'USDARS', 'USDNGN', 'USDPHP', 'USDTRY', 'USDRUB', 'USDINR', 'USDCNY']
currencies_names = ['Brazilian Real - USDBRL', 'Mexican Peso - USDMXN', 'Colombian Peso - USDCOP', 'Argentine Peso - USDARS', 'Nigerian Naira - USDNGN',
                  'Philippine Peso - USDPHP', 'Turkish Lira - USDTRY', 'Russian Ruble - USDRUB', 'Indian Rupee - USDINR', 'Chinese Yuan - USDCNY']
dataframes = []

for i in Currencies:
    data_X = yf.download(i+'=X', start=start_date, end=end_date)
    dataframes.append(data_X)#
```
## 2. Data Preparation
```python
# Define cleaning functions:

# Add new column with the Close change percent between rows
def add_pct_change(df):
    if any(col in df.columns for col in set(['Close_Change_Pct', 'Close_Change_Pct_x', 'Close_Change_Pct_y'])):
        df.drop(['Close_Change_Pct'], axis=1, inplace=True, errors='ignore')
        df['Close_Change_Pct'] = df['Close'].pct_change()
    else:
        df['Close_Change_Pct'] = df['Close'].pct_change()
    return df.sort_values(by='Close_Change_Pct', ascending=True).head(10) # Sort the dates with the largest change pct

# Where the Close change pct is <= -70% it is replaced to the previous value
def replace_close(df):
    for row in range(0,len(df)):
        df['Close'] = np.where((df['Close_Change_Pct'] <= -0.7), df['Close'].shift(1), df['Close'])

# Feature Creation: Create time series features per period
def Feature_Creation(df):
    df.drop(['CloseScaled','DayOfWeek','Month','Quarter','Year','Prediction'], axis=1, inplace=True, errors='ignore')
    df['CloseScaled'] = MinMaxScaler().fit_transform(df.filter(['Close']).values)
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Year'] = df.index.year
    return df

# Apply cleaning functions
for df in dataframes:
    add_pct_change(df)
    replace_close(df)
    replace_close(df)
    Feature_Creation(df)

# Visualize Cleaned Data

# Create a figure with 10 subplots arranged in a 5x2 grid
fig, axes = plt.subplots(2,5, figsize=(20,10))
plt.suptitle("Historic Closing Exchange Rates - Cleaned Data").set_y(1)

for df, ax, currency, name in zip(dataframes, axes.flatten(), Currencies, currencies_names):
    ax.plot(df['Close'])
    plt.gcf().autofmt_xdate()
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(name)
    ax.grid(True)
plt.tight_layout()
```

![Figure 1: Historic Closing Exchange Rates - Cleaned Data.](https://)



