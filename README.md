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

![Figure 3.1.1: XGB Regressor](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-23%20203141.png)
![Figure 3.1.2: XGB Regressor](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-23%20203159.png)
![Figure 3.1.3: XGB Regressor](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-23%20203218.png)
![Figure 3.1.4: XGB Regressor](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-23%20203236.png)
![Figure 3.1.5: XGB Regressor](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-23%20203259.png)

### 3.2 Long Short Term Memory (LSTM)
```python
# Define LSTM functions:
def LSTM_Model(data):

    # Apply cleaning functions on data
    add_pct_change(data)
    replace_close(data)
    replace_close(data)
    Feature_Creation(data)

    # Training Dataset
    split_date = int(len(data) * 0.7)
    train = np.array(data.CloseScaled.iloc[ :split_date]) # Train in 70% of first dates
    X_train = []
    y_train = []
    for i in range(60, split_date):
        X_train.append(train[i-60:i])
        y_train.append(train[i])
    X_train, y_train= np.array(X_train), np.array(y_train) # convert the train data into array
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # Reshape the data

    # Testing Dataset
    test = np.array(data.CloseScaled.iloc[split_date: ]) # Test in 30% after split
    X_test = []
    y_test = data.Close.iloc[split_date+60: ] #normal values from original data
    for i in range(60, len(test)):
        X_test.append(test[i-60:i])
    X_test = np.array(X_test) # convert the train data into array
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))#Reshape the data

    # Create model LSTM
    seq = Sequential() # Initializing the RNN
    seq.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1))) # Adding the first LSTM layer
    seq.add(LSTM(50, return_sequences=False)) # Adding the Second LSTM layer
    seq.add(Dense(25))
    seq.add(Dense(1))
    seq.compile(optimizer='adam', loss='mean_squared_error')# Compile the model
    seq.fit(X_train, y_train, batch_size=32, epochs=1)# Traing the model. Set the epochs=10 takes 10 minutes (100 takes too long)

    # Get model predicted values
    scaler = MinMaxScaler()
    scaler.fit(data.filter(['Close']).values)
    pred = seq.predict(X_test)
    pred = scaler.inverse_transform(pred) # "inverse scaled values to original values"

    # Calculate the mean squared error on the training data
    mse_seq = mean_squared_error(y_test, pred)
    rmse_seq = sqrt(mse_seq)
    print(f'{Currency} RMSE: {rmse_seq:.2f}')

    # Split Close non-scaled data into train and valid df
    train_df = pd.DataFrame(data.Close.iloc[ :split_date+60]) # Train in 70% of first dates
    valid_df = pd.DataFrame(data.Close.iloc[split_date+60: ]) # Test in 30% after split
    valid_df['Prediction'] = pred # Add Predictions column with the inverse scaled values

    # Plot the Training and Testing data sets and zoom in
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
    plt.suptitle("LSTM Sequential Model").set_y(1)
    train_df.plot(ax=ax1, label='Training Set', title=f'Raw and Prediction Data - {Currency}')
    valid_df.plot(ax=ax1, label=['Valid Data','Prediction Data']).grid(True)
    ax1.axvline(data.index[split_date+60], color='grey', ls='--')
    ax1.legend(['Training Data', 'Valid Data','Prediction Data'])
    valid_df['Close'].plot(ax = ax2, color='darkseagreen', title=f'Zoom in Test Raw and Prediction Data - {Currency}')
    valid_df['Prediction'].plot(ax = ax2, style='--', color='red').grid(True)
    ax2.axvline(data.index[split_date+60], color='grey', ls='--')
    ax2.legend(['Valid Data', 'Prediction Data'])
    plt.tight_layout()

    return 

# Perform LSTM across all currencies
for df, Currency in zip(dataframes, Currencies):
    LSTM_Model(df)
```

![Figure 3.2.1: LSTM](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-23%20214453.png)
![Figure 3.2.2: LSTM](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-23%20214510.png)
![Figure 3.2.3: LSTM](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-23%20214530.png)
![Figure 3.2.4: LSTM](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-23%20214548.png)
![Figure 3.2.5: LSTM](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-23%20214608.png)

### 3.3 Support Vector Classifier (SVC)
```python
# Define SVC functions:
def SVC_Model(data):

    # Apply cleaning functions
    add_pct_change(data)
    replace_close(data)
    replace_close(data)

    # Create independent and dependent variables
    data['High-Low'] = data['High'] - data['Low']
    data['Open-Close'] = data['Open'] - data['Close']
    X = data[['Open-Close', 'High-Low', 'Close']] # Define independent variables
    # Create signals: If tomorrow's close is > today's, then 1 increase, 0 otherwise
    y = np.where(data.Close.shift(-1) > data.Close, 1, 0) # Target variable

    # Training Dataset
    split_date = int(len(data) * 0.9)
    X_train = X[:split_date]
    y_train = y[:split_date]
    # Testing Dataset
    X_test = X[split_date:]
    y_test = y[split_date:]

    # Create the model SVC
    svc = SVC()

    svc.fit(X_train[['Open-Close', 'High-Low']],y_train)# Traing the model
    svc_score_train = svc.score(X_train[['Open-Close', 'High-Low']],y_train)# score of the model on Train
    svc_score_test = svc.score(X_test[['Open-Close', 'High-Low']],y_test)# score of the model on Test

    data['Predictions'] = svc.predict(X[['Open-Close', 'High-Low']])# model predictions
    data['Return'] = data['Close'].pct_change() # Calculate daily returns
    data['Strat_Return'] = data['Predictions'].shift(1)*data['Return'] # Calculate strategy returns
    data['Cumul_Return'] = data['Return'].cumsum() # Calculate cumulative returns
    data['Cumul_Strat'] = data['Strat_Return'].cumsum() # Calculate strategy returns

    return

# Perform SVC across all currencies

fig, axes = plt.subplots(2,5, figsize=(20,10))
plt.suptitle("SVC Model").set_y(1)

for df, ax, currency, name in zip(dataframes, axes.flatten(), Currencies, currencies_names):
    SVC_Model(df)
    ax.plot(df['Cumul_Return'], label='Currency Returns')
    ax.plot(df['Cumul_Strat'], label='Strategy Returns')
    plt.gcf().autofmt_xdate()
    ax.set_title(name)
    ax.grid(True)
    ax.legend()
plt.tight_layout()
```

![Figure 3.3.1: SVC](https://github.com/cvas91/Forecasting_Currencies/blob/main/Figures/Screenshot%202023-07-20%20172642.png?raw=true)

### 3.4 Random Forest Regressor (RFR)

```python
# Define RFR functions:
def RFR_Model(data,Currency):

    # Split the dataset
    X = data[['Open','High','Low']]
    X_train = X[ :len(data)-1] # all rows but not the last one
    X_test = X.tail(1) # the last one row
    y = data['Close']
    y_train = y[ :len(data)-1] # all rows but not the last one
    y_test = y.tail(1) # the last one row

    # Create the model Random Forest Regressor
    RFR = RandomForestRegressor()

    # Train the model
    RFR.fit(X_train,y_train)

    # Test the model
    predictions = RFR.predict(X_train)

    # Make prediction
    prediction = RFR.predict(X_test)
    print(Currency,'prediction:')
    print('RFR score is:', (RFR.score(X_train,y_train)*100).round(3),'%')
    print('RFR predicts the last day to be:', prediction.round(3))
    print('Actual value is:',y_test.values[0].round(3)) # this should be the last value from the data imported
    print('Difference between actual and predicted is:',(y_test.values[0] - prediction).round(3))
    print()

    return
```

### 3.5 Evaluating Models with Pycaret
```python
# Define Pycaret functions:
def Comparing_Model(data):

    # Apply cleaning functions on data
    add_pct_change(data)
    replace_close(data)
    replace_close(data)

    future_days = 10 # variable for predicting days out into the future
    data['Future_Price'] = data['Close'].shift(-future_days) # create a new column for the target feature shifted 'n days' up

    X = data[['Close','Future_Price']]
    X = X[ :len(data)-future_days] # all rows but not the future days
    y = data['Future_Price']
    y = y[ :-future_days] # all rows but not the future days
    split_date = int(len(data) * 0.7) # Change the % to train data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_date, random_state=0, shuffle=False) 

    print('\n \033[1m' + Currency + '\033[0m')
    regression_setup = setup(data=X_train, target='Future_Price', session_id=123) # Initialize the setup
    comparing = compare_models(sort='RMSE') # Also sort by RMSE?descending??
    best_model = create_model(comparing) # The best model for each currency
    unseen_predictions = predict_model(best_model, data=X_test)
    data = pd.merge(data, unseen_predictions['prediction_label'], how='left', left_index=True, right_index=True)

    # Visualize Scaled / Predictions and Zoom in
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
    plt.suptitle("Best Model").set_y(1)
    data['Close'].plot(ax=ax1, title=f'Scaled and Prediction Datasets - {Currency}')
    data[['prediction_label']].plot(ax=ax1, style='--', color='red').grid(True)
    ax1.axvline(data.index[split_date], color='grey', ls='--')
    ax1.legend(['Raw Data', 'Prediction Data'])
    data.loc[data.index >= data.index[split_date]]['Close'].plot(ax=ax2, title=f'Zoom in Test Raw and Prediction Dataset - {Currency}')
    data.loc[data.index >= data.index[split_date]]['prediction_label'].plot(ax=ax2, style='--', color='red').grid(True)
    ax2.axvline(data.index[split_date], color='grey', ls='--')
    ax2.legend(['Raw Data', 'Prediction Data'])
    plt.tight_layout()

    return 

# Perform Pycaret evaluating models across all currencies
for df, Currency in zip(dataframes, Currencies):
     Comparing_Model(df)
plt.show()
```

![Figure 3.5.1: LSTM]()

