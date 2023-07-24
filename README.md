# Forecasting Emerging Currencies Exchange Rates Applying Machine Learning Techniques

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

# Visualize Feature / Target Relationship
# Horizontal plots
fig, axes = plt.subplots(2,5, figsize=(20,10))
plt.suptitle("Closing Rate by Year").set_y(1)

for df, ax, currency, name in zip(dataframes, axes.flatten(), Currencies, currencies_names):
    sns.boxplot(y='Close', x= 'Year', data=df, ax=ax, orient='v').set_title(name)#
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # x_ticks, x_labels = plt.xticks()# Get the current x tick labels and positions
    # plt.xticks(x_ticks[::2], x_labels[::2])# Set the x tick labels and positions to only include every other label
    ax.grid(True)
plt.tight_layout()
```
![Figure 1: Historic Closing Exchange Rates - Cleaned Data.](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-23%20195652.png)

![Figure 2: Closing Rate by Year.](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Closing%20Rate%20by%20Year.png)

## 3. Machine Learning Models
### 3.1 Extreme Gradient Boosting XGB Regressor

```python
# Define XGB functions:
def XGB_Model(data):
    # Apply cleaning functions on data
    add_pct_change(data)
    replace_close(data)
    replace_close(data)
    Feature_Creation(data)

    train_df = pd.DataFrame(data.CloseScaled.iloc[ :split_date]) # Train in 70% of first dates
    test_df = pd.DataFrame(data.CloseScaled.iloc[split_date: ]) # Test in 30% after split

    X_train_df = data[['DayOfWeek', 'Month', 'Quarter', 'Year']].iloc[ :split_date]
    y_train_df = data[['CloseScaled']].iloc[ :split_date]
    X_test_df = data[['DayOfWeek', 'Month', 'Quarter', 'Year']].iloc[split_date: ]
    y_test_df = data[['CloseScaled']].iloc[split_date: ]

    reg = xgb.XGBRegressor(n_estimators = 1000, early_stopping_rounds =50, learning_rate = 0.01)
    reg.fit(X_train_df, y_train_df, eval_set=[(X_train_df, y_train_df), (X_test_df, y_test_df)], verbose=100)

    test_df['Prediction'] = reg.predict(X_test_df) # Add the predictions in a new column

    # Merge the predictions with the initial df
    if any(col in data.columns for col in set(['Prediction', 'Prediction_x', 'Prediction_y'])):
        data.drop(['Prediction','Prediction_x','Prediction_y'], axis=1, inplace=True, errors='ignore')
        data = data.merge(test_df['Prediction'], how='left', left_index=True, right_index=True)
    else:
        data = data.merge(test_df['Prediction'], how='left', left_index=True, right_index=True)

    RMSE = np.sqrt(mean_squared_error(test_df['CloseScaled'], test_df['Prediction']))
    print(f'{Currency} - RMSE Score on Test Set: {RMSE: 0.3f}') # This should be the same score as validation_1-rmse

    # Optimized visuals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
    plt.suptitle("XGBRegressor Model").set_y(1)
    axes = [ax1, ax2]
    titles = [f'Scaled and Prediction Data - {Currency}', f'Zoom in Test Raw and Prediction Data - {Currency}']
    data_to_plot = [data[['CloseScaled', 'Prediction']], data.loc[data.index >= data.index[split_date], ['CloseScaled', 'Prediction']]]
    for ax, title, data_to in zip(axes, titles, data_to_plot):
        data_to['CloseScaled'].plot(ax=ax, title=title)
        data_to['Prediction'].plot(ax=ax, style='--', color='red').grid(True)
        ax.axvline(data.index[split_date], color='grey', ls='--')
        ax.legend(['Raw Data', 'Prediction Data'])
        plt.tight_layout()

    return 

# Perform XGB across all currencies
for df, Currency in zip(dataframes, Currencies):
    XGB_Model(df)
```

![Figure 3: XGB Regressor](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-23%20203141.png)
