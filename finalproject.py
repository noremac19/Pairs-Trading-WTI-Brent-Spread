import numpy as np
import pandas as pd
from arch import arch_model
from copulas.univariate import GaussianKDE
from copulas.multivariate import GaussianMultivariate 
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from scipy.stats import zscore

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Data Preprocessing 
brent_futures = pd.read_csv('BrentData.csv')
wti_futures = pd.read_csv('WTIData.csv')

brent_futures = brent_futures[['Date', 'PX_LAST', 'FUT_CUR_GEN_TICKER']]
wti_futures = wti_futures[['Date', 'PX_LAST', 'FUT_CUR_GEN_TICKER']]
brent_futures

brent_futures['Date'] = pd.to_datetime(brent_futures['Date'])
wti_futures['Date'] = pd.to_datetime(wti_futures['Date'])

brent_futures.columns = ['Date', 'Brent', 'Active_Brent']
brent_futures.set_index('Date')
brent_futures.dropna(inplace=True)

wti_futures.columns = ['Date', 'WTI', 'Active_WTI']
wti_futures.set_index('Date')
wti_futures.dropna(inplace=True)

data = pd.merge(wti_futures, brent_futures, how='outer')

data.set_index('Date', inplace=True)

data.sort_index(ascending=True, inplace=True)

data['WTI'] = data['WTI'].fillna(method='ffill')
data['Brent'] = data['Brent'].fillna(method='ffill')

data['Spread'] = data['WTI'] - data['Brent']

data = data['2021-01-01':]

# Step 1: Regress WTI on Brent
model = sm.OLS(data['WTI'], sm.add_constant(data['Brent']))
result = model.fit()
data['residuals'] = result.resid

# Step 2: Perform the Augmented Dickey-Fuller test on the residuals
adf_test = adfuller(data['residuals'])

# Print the results
print("ADF Statistic:", adf_test[0])
print("p-value:", adf_test[1])
print("Critical Values:")
for key, value in adf_test[4].items():
    print(f"   {key}: {value}")

# Interpretation
if adf_test[1] < 0.05:
    print("The residuals are stationary. WTI and Brent prices are cointegrated.")
else:
    print("The residuals are not stationary. WTI and Brent prices are not cointegrated.")

# Plot WTI and Brent prices
plt.figure(figsize=(14, 7))

plt.plot(data.index, data['WTI'], label='WTI Price')
plt.plot(data.index, data['Brent'], label='Brent Price')

# Customize the plot
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('WTI vs Brent Prices Over Time')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Plot the spread over time
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Spread'], label='WTI-Brent Spread')

# Customize the plot
plt.xlabel('Date')
plt.ylabel('Spread (WTI - Brent)')
plt.title('WTI/Brent Spread Over Time')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Step 1: Load data and calculate returns
data['Log_Returns_Brent'] = np.log(data['Brent']).diff()
data['Log_Returns_WTI'] = np.log(data['WTI']).diff()
data.dropna(inplace=True)

data['Log_Returns_Brent'] = data['Log_Returns_Brent'] * 100
data['Log_Returns_WTI'] = data['Log_Returns_WTI'] * 100

data['Returns_Spread'] = data['Log_Returns_WTI'] - data['Log_Returns_Brent']

# Step 2: Fit GARCH models for marginal distributions
def fit_garch(returns):
    model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = model.fit(disp='off')
    return garch_fit.resid / garch_fit.conditional_volatility

residuals_brent = fit_garch(data['Log_Returns_Brent'])
residuals_wti = fit_garch(data['Log_Returns_WTI'])

kde_wti = GaussianKDE()
kde_wti.fit(residuals_wti)
wti_uniform = kde_wti.cdf(residuals_wti)  # Transforms to uniform [0, 1]

kde_brent = GaussianKDE()
kde_brent.fit(residuals_brent)
brent_uniform = kde_brent.cdf(residuals_brent)  # Transforms to uniform [0, 1]

# Step 4: Specify and fit copula model with clipped values to improve stability
wti_uniform = np.clip(wti_uniform, 0.001, 0.999)
brent_uniform = np.clip(brent_uniform, 0.001, 0.999)

# Combine data for copula model
copula_data = np.column_stack([wti_uniform, brent_uniform])
copula_model = GaussianMultivariate()
copula_model.fit(pd.DataFrame(copula_data, columns=['WTI', 'Brent']))

# Step 5: Implement and fit regime-switching model (HMM for two states)
data['Spread_z'] = zscore(data['Returns_Spread'])
X = data[['Spread_z']].values
hmm_model = GaussianHMM(n_components=2, covariance_type="diag", random_state=42)
hmm_model.fit(X)
regimes = hmm_model.predict(X)

# Map regimes to high and low dependency states
# Assuming regime 0 is 'High' if it has a higher mean of Spread_z
regime_means = [data['Spread_z'][regimes == i].mean() for i in range(hmm_model.n_components)]
high_dep_regime = np.argmax(regime_means)  # Index of regime with higher mean
data['Regime'] = np.where(regimes == high_dep_regime, 'High', 'Low')

# Step 7: Generate Trading Signals Based on Regime and Spread Z-score
# def trading_signal(row):
#     if row['Regime'] == 'High':  # Trade only in high dependency regime
#         if row['Spread_z'] > 1.5:
#             return 'Sell Spread'  # Go short
#         elif row['Spread_z'] < -1.5:
#             return 'Buy Spread'  # Go long
#     return 'Hold'

def trading_signal(row):
    if row['Regime'] == 'High':  # High dependency regime
        if row['Spread_z'] > 2:
            return 'Sell Spread'  # Go short
        elif row['Spread_z'] < -2:
            return 'Buy Spread'  # Go long
    elif row['Regime'] == 'Low':  # Low dependency regime
        if row['Spread_z'] > 1:
            return 'Sell Spread'
        elif row['Spread_z'] < -1:
            return 'Buy Spread'
    return 'Hold'


data['Signal'] = data.apply(trading_signal, axis=1)

# Step 9: Backtesting the Strategy
# Initialize portfolio parameters
initial_cash = 10000000  # Starting cash in dollars
cash = initial_cash
portfolio_value = [initial_cash]  # Start with initial cash
open_position = None  # Track open position details
contract_size = 1000  # Standard contract size in barrels
wti_transaction_cost = 0.70
brent_transaction_cost = 0.85
data['PnL'] = 0  # To store realized PnL
data['Transaction_Costs'] = 0 # To store transaction costs for each trade
# Define function to calculate contracts based on cash and price
def calculate_contracts(price, cash):
    return int((cash / 2) / (price * contract_size))  # Divide cash by 2 for each leg

# Backtesting loop
for i in range(1, len(data)):
    signal = data['Signal'].iloc[i]
    wti_price = data['WTI'].iloc[i]
    brent_price = data['Brent'].iloc[i]

    # Daily PnL calculation
    if open_position:
        if open_position['type'] == 'Buy Spread':
            daily_pnl_wti = (wti_price - open_position['wti_entry']) * open_position['wti_contracts'] * contract_size
            daily_pnl_brent = (open_position['brent_entry'] - brent_price) * open_position['brent_contracts'] * contract_size
        elif open_position['type'] == 'Sell Spread':
            daily_pnl_wti = (open_position['wti_entry'] - wti_price) * open_position['wti_contracts'] * contract_size
            daily_pnl_brent = (brent_price - open_position['brent_entry']) * open_position['brent_contracts'] * contract_size
        
        # Aggregate PnL for the spread position
        daily_pnl = daily_pnl_wti + daily_pnl_brent
        cash += daily_pnl
        # print(cash)
        # print(i)
        # print(open_position)
        # print(daily_pnl)
        # print('-------------')
        data['PnL'].iloc[i] = daily_pnl  # Update PnL in the DataFrame
        data['Transaction_Costs'].iloc[i] = open_position['transaction_costs'] * 2

    # Signal handling
    if signal == 'Hold':
        if open_position:
            open_position = None  # Reset the position

        continue
        
    elif signal == 'Buy Spread' and (not open_position or open_position['type'] == 'Sell Spread'):
        # Close existing position if it's a "Sell Spread"
        if open_position and open_position['type'] == 'Sell Spread':
            #cash = open_position['wti_entry'] * open_position['wti_contracts'] * contract_size + brent_price * open_position['brent_contracts'] * contract_size
            open_position = None

        # Open a new "Buy Spread" position (Long WTI, Short Brent)
        wti_contracts = calculate_contracts(wti_price, cash)
        brent_contracts = calculate_contracts(brent_price, cash)
        transaction_costs = wti_contracts * wti_transaction_cost + brent_contracts * brent_transaction_cost
        open_position = {
            'type': 'Buy Spread',
            'wti_entry': wti_price,
            'brent_entry': brent_price,
            'wti_contracts': wti_contracts,
            'brent_contracts': brent_contracts,
            'transaction_costs': transaction_costs
        }

    elif signal == 'Sell Spread' and (not open_position or open_position['type'] == 'Buy Spread'):
        # Close existing position if it's a "Buy Spread"
        if open_position and open_position['type'] == 'Buy Spread':
            #cash = wti_price * open_position['wti_contracts'] * contract_size + open_position['brent_entry'] * open_position['brent_contracts'] * contract_size
            open_position = None

        # Open a new "Sell Spread" position (Short WTI, Long Brent)
        wti_contracts = calculate_contracts(wti_price, cash)
        brent_contracts = calculate_contracts(brent_price, cash)
        transaction_costs = wti_contracts * wti_transaction_cost + brent_contracts * brent_transaction_cost
        open_position = {
            'type': 'Sell Spread',
            'wti_entry': wti_price,
            'brent_entry': brent_price,
            'wti_contracts': wti_contracts,
            'brent_contracts': brent_contracts,
            'transaction_costs': transaction_costs
        }

        
data['Cumulative_PnL'] = data['PnL'].cumsum()
data['Cumulative_Transaction_Costs'] = data['Transaction_Costs'].cumsum()
    # Calculate portfolio value: cash + open position value (if any)
#     position_value = 0
#     if open_position:
#         if open_position['type'] == 'Buy Spread':
#             position_value = (brent_price - open_position['brent_entry']) * open_position['brent_contracts'] * contract_size + (open_position['wti_entry'] - wti_price) * open_position['wti_contracts'] * contract_size
#         elif open_position['type'] == 'Sell Spread':
#             position_value = (open_position['brent_entry'] - brent_price) * open_position['brent_contracts'] * contract_size + (wti_price - open_position['wti_entry']) * open_position['wti_contracts'] * contract_size
    
#     # Add cash and position value to get the current portfolio value
#     total_portfolio_value = cash + position_value
#     portfolio_value.append(total_portfolio_value)

# Add portfolio values to DataFrame
# data['Portfolio Value'] = portfolio_value

plt.figure(figsize=(12, 6))
plt.plot(data['Spread_z'], label='Spread_z')
plt.title('Z-Score of Spread Over Time')
plt.xlabel('Time')
plt.ylabel('Spread Z-Score')
plt.legend()
plt.grid(True)
plt.show()

# Step 1: Calculate Portfolio Value and Daily Returns
data['Portfolio_Value'] = data['PnL'].cumsum() + initial_cash
data['Daily_Return'] = data['Portfolio_Value'].pct_change()

# Step 2: Mean Return and Standard Deviation
mean_return = data['Daily_Return'].mean()
std_dev = data['Daily_Return'].std()

# Step 3: Risk-Free Rate
risk_free_rate = 0.03  # 3% annual risk-free rate
daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1

# Step 4: Sharpe Ratio
sharpe_ratio = (mean_return - daily_risk_free_rate) / std_dev

# Step 5: Annualized Sharpe Ratio (optional)
annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)

print(f"Sharpe Ratio (Daily): {sharpe_ratio}")
print(f"Sharpe Ratio (Annualized): {annualized_sharpe_ratio}")
