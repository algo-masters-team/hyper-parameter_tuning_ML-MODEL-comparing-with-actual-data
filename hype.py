import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta
import yfinance as yf
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

def fetch_and_process_data(ticker):
    data = yf.download(ticker)
    data.drop('Adj Close', axis=1, inplace=True)
    data['returns'] = data['Close'].pct_change(1)
    data['lastclose'] = data['Close'].shift(-1)
    data['upper'] = data['Close'].rolling(window=96).max().shift(1)
    data['lower'] = data['Close'].rolling(window=96).min().shift(1)
    data['middle'] = (data['upper'] + data['lower']) / 2
    data['SMA200'] = ta.SMA(data['Close'], timeperiod=200)
    data['signal_BASEDONDC&SMA_buy'] = np.where((data['Close'] > data['upper']) & (data['Close'] > data['SMA']), 1, 0)
    data['signal_BASEDONDC&SMA_sell'] = np.where((data['Close'] < data['lower']) & (data['Close'] < data['SMA']), -1, 0)
    data['signal'] = data['signal_BASEDONDC&SMA_buy'] + data['signal_BASEDONDC&SMA_sell']
    data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data.dropna(inplace=True)
    return data

# Fetch and process data for training stocks
data_pageind = fetch_and_process_data('PAGEIND.NS')
data_reliance = fetch_and_process_data('RELIANCE.NS')

# Fetch and process data for testing stocks
data_tcs = fetch_and_process_data('TCS.NS')
data_infy = fetch_and_process_data('INFY.NS')

# Combine training data
train_data = pd.concat([data_pageind, data_reliance])
X_train = train_data[['lastclose', 'upper', 'lower', 'middle', 'SMA200', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']]
y_train = train_data['returns']

# Combine testing data
test_data = data_tcs
X_test = test_data[['lastclose', 'upper', 'lower', 'middle', 'SMA200', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']]
y_test = test_data['returns']

# Models and hyperparameter grids
models = {
    'Linear Regression': (LinearRegression(), {}),
    'Decision Tree': (DecisionTreeRegressor(random_state=42), {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]}),
    'Random Forest': (RandomForestRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [5, 10, 20]}),
    'Gradient Boosting': (GradientBoostingRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
    'SVR': (SVR(), {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}),
    'K Neighbors': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}),
    'MLP': (MLPRegressor(max_iter=1000, random_state=42), {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh']}),
    'Bayesian Ridge': (BayesianRidge(), {})
}

# Train models with hyperparameter tuning
best_models = {}
for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f'Best parameters for {name}: {grid.best_params_}')

# Predictions
for name, model in best_models.items():
    test_data[f'predict_{name}'] = model.predict(X_test)

# Plot predictions
test_data[[f'predict_{name}' for name in best_models]].plot()
plt.title('Stock Return Predictions')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend(best_models.keys())
plt.show()

# Strategy returns
for name in best_models:
    test_data[f'position_{name}'] = np.sign(test_data[f'predict_{name}'])
    test_data[f'strategy_{name}'] = test_data['returns'] * test_data[f'position_{name}'].shift(1)

# Plot cumulative strategy returns
cumulative_returns = (test_data[[f'strategy_{name}' for name in best_models]].cumsum() * 100)
cumulative_returns.plot()
plt.title('Cumulative Strategy Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns (%)')
plt.legend(best_models.keys())
plt.show()

# Performance evaluation
def calculate_performance(position_column, returns):
    no_of_wins = ((returns[position_column] == 1) & (returns['returns'].shift(-1) > 0)).sum()
    no_of_losses = ((returns[position_column] == 1) & (returns['returns'].shift(-1) <= 0)).sum()
    no_of_noreturn = (returns[position_column] == 0).sum()
    total_trades = no_of_wins + no_of_losses + no_of_noreturn
    win_rate = (no_of_wins / total_trades) * 100 if total_trades > 0 else 0
    error_rate = (no_of_losses / total_trades) * 100 if total_trades > 0 else 0
    return no_of_wins, no_of_losses, no_of_noreturn, win_rate, error_rate

# Print performance metrics
for name in best_models:
    performance = calculate_performance(f'position_{name}', test_data)
    print(f'{name} Model:\nNumber of wins: {performance[0]}\nNumber of losses: {performance[1]}\nNumber of no returns: {performance[2]}\nWin rate: {performance[3]:.2f}%\nError rate: {performance[4]:.2f}%\n')

print(test_data)
