###########
# Imports #
###########
import os, pickle
os.chdir('E:/DARWIN_API_TUTORIALS/PYTHON/')

from MINIONS.dwx_graphics_helpers import DWX_Graphics_Helpers
from plotly.offline import init_notebook_mode
from scipy.stats import zscore
import pickle, warnings
import pandas as pd
import numpy as np

################################
# Some configuration for later #
################################
warnings.simplefilter("ignore") # Suppress warnings
init_notebook_mode(connected=True)
################################

# Create DWX Graphics Helpers object for later
_graphics = DWX_Graphics_Helpers()

# Load DataFrame of DARWIN quotes (Daily precision) from pickle archive.
quotes = pickle.load(open('../DATA/jn_all_quotes_active_deleted_12062019.pkl', 'rb'))

# Remove non-business days (consider Monday to Friday only)
quotes = quotes[quotes.index.dayofweek < 5]

# Load FX Market Volatility data upto 2023-04-23-23.09.2023 (for evaluation later)
voldf = pd.read_csv('../DATA/volatility.beginning.to.2023-04-2023-23709-2023.csv', 
                    index_col=0, 
                    parse_dates=True,
                    infer_datetime_format=True)

# Print DARWINs in dataset
print(f'DARWIN assets in dataset: {quotes.shape[1]}') 


print(f'\nDARWIN Symbols: \n{quotes.columns}')


test_darwin = 'LVS.4.20'


_graphics._plotly_dataframe_scatter_(
                            _df=pd.DataFrame(quotes[test_darwin].dropna()),
                            _x_title = "Date / Time",
                            _y_title = "Quote",
                            _main_title = f'${test_darwin} Quotes',
                            _plot_only = True)

print(quotes[test_darwin].tail())
pass; 

def calculate_log_returns(quotes):
    return np.log(quotes) - np.log(quotes.shift(1))

def calculate_simple_returns(quotes):
    return quotes.pct_change()
# Calculate both simple and log returns for the loaded DataFrame above
log_returns = calculate_log_returns(quotes)
simple_returns = calculate_simple_returns(quotes)

# Print shape
print(f'\nShape of Returns: {log_returns.shape}') # 3204 rows, 5331 DARWINs

# Print last 5 log returns of randomly chosen DARWIN from earlier
print(f'\nLast 5 log returns of example DARWIN {test_darwin}: \
\n{log_returns.loc[:,test_darwin].tail(5)}')

# Print last 5 simple returns of randomly chosen DARWIN from earlier
print(f'\nLast 5 simple returns of example DARWIN {test_darwin}: \
\n{simple_returns.loc[:,test_darwin].tail(5)}')

# Print and plot all-time return of test DARWIN
print(f'\n${test_darwin} Price-based return: \
{quotes[test_darwin].dropna().values[-1] / quotes[test_darwin].dropna().values[0] - 1}')

# Compounded log return
print(f'\n${test_darwin} Compounded log return: {log_returns[test_darwin].dropna().sum()}')
print(f'\n${test_darwin} Compounded log return (converted to simple): \
{np.exp(np.log(quotes[test_darwin].dropna().values[-1]) - np.log(quotes[test_darwin].dropna().values[0])) - 1}')

# Compounded simple return
print(f'\n${test_darwin} Compounded simple return: {((1 + simple_returns[test_darwin].dropna()).cumprod() - 1)[-1]}')

def shifted_returns(log_returns, periods=1):
    return log_returns.shift(periods)


def get_top_n_in_row(row, _n):
    
    top_n = row[list(np.nonzero(row.values)[0])].nlargest(_n)
    _out = pd.Series(data=0, index=row.index)
    
    if len(top_n) == _n and top_n.values.min() > 0:
        _out[list(top_n.index)] = 1
    
    # Default
    return _out
    
def get_top_n_darwins(past_returns, n=20):
    return past_returns.apply(lambda row: get_top_n_in_row(row, n), axis=1)


past_returns = shifted_returns(log_returns, 1)
future_returns = shifted_returns(log_returns, -1)

# Plot and print last 21 trading days of Quotes, past and future returns for the test DARWIN
df_c_test = pd.concat([quotes[test_darwin],
                 log_returns[test_darwin],
                 past_returns[test_darwin],
                 future_returns[test_darwin]
                ], axis=1)

# Set meaningful columns
df_c_test.columns = ['Quote','log_return','past_return','future_return']

# Plot
_graphics._plotly_dataframe_scatter_(
                            _df=pd.DataFrame(df_c_test.iloc[-21:, 1:].dropna()),
                            _x_title = "Date / Time",
                            _y_title = "Returns",
                            _main_title = f'${test_darwin} Log, Past & Future Returns',
                            _plot_only = True)

# Print last 5 log, past and future returns
print(f'Last 5 log, past and future return for ${test_darwin}:\n{df_c_test.tail(5)}')

# Let's see what the output of get_top_n_darwins() looks like.
top_20_darwins = get_top_n_darwins(past_returns, 20)

pass;

# Example: Top 20 DARWINs to buy next
print(f'\nTop 20 DARWINs symbol names for next day:\n\
{top_20_darwins.iloc[-1,:].sort_values(ascending=False).index[:20].values.tolist()}')

# Example: Top 20 DARWINs bought historically
print(f'\nTop 20 DARWINs traded the most, historically:\n\
{top_20_darwins.sum().sort_values(ascending=False).index[:20].values.tolist()}')

def calculate_strategy_returns(top_darwins_df, future_returns_df, n=20, cost=np.log(1 + 0.002)):
    return top_darwins_df * (future_returns_df - cost) / n


def darwin_momentum_strategy(_timeframe='', top_n=50, _tcost=0.002):
    
    # Load DataFrame of DARWIN quotes (Daily precision) from pickle archive.
    quotes = pickle.load(open('../DATA/jn_all_quotes_active_deleted_12062019.pkl', 'rb'))

    # Remove non-business days (consider Monday to Friday only)
    quotes = quotes[quotes.index.dayofweek < 5]
    
    # Resample if timeframe != ''
    if _timeframe == 'W':
        quotes = quotes.resample('W-FRI').last()
    elif _timeframe == 'M':
        quotes = quotes.resample('M').last()
      
    # Calculate log, past and future returns
    log_returns = calculate_log_returns(quotes)
    past_returns = shifted_returns(log_returns, 1)
    future_returns = shifted_returns(log_returns, -1)
    
    # Generate DataFrame of Top n DARWINs by periodic return
    top_n_darwins = get_top_n_darwins(past_returns, top_n)
    
    # Calculate strategy returns
    strategy_returns = calculate_strategy_returns(top_n_darwins, future_returns, top_n, np.log(1 + _tcost))

    # Plot strategy returns
    strategy_returns = strategy_returns.sum(axis=1)
    cumulative_strategy_returns = strategy_returns.cumsum()
    cumulative_strategy_returns[cumulative_strategy_returns < -1] = -1

    return [strategy_returns, cumulative_strategy_returns]


def plot_strategy_results(_timeframe='', 
                          top_n_range=[10,101,10], 
                          _tcost=0.002,
                          return_results=True):
    
    results = {}
    
    for _n in range(top_n_range[0],top_n_range[1],top_n_range[2]):
        print(f'Processing {_n} DARWINs..')
        results[(_timeframe,_n,_tcost)] = darwin_momentum_strategy(_timeframe, _n, _tcost)

    # Transform results into iterable data structures
    _k = list(results.keys())
    _v = list(results.values())
    
    print('\nGenerating plot.. please wait..')
    _graphics._plotly_dataframe_scatter_(
                            _df=pd.DataFrame(data={str(_k[i]): _v[i][1].values.tolist() 
                                                   for i in range(len(_k))},
                                             index=_v[0][1].index),
                            _x_title = 'Date / Time',
                            _y_title = 'Returns',
                            _main_title = f'[Strategy] Timeframe: {_timeframe}, Cost: {_tcost*100}%, Top "n" DARWINs = {top_n_range}',
                            _plot_only = True)
    
    print('\n..DONE!')
    
    if return_results:
        return [_k,_v]
    
# This dictionary will hold strategy results for each combination employed
results = {}


# Daily timeframe, 10 to 100 DARWINs in steps of 5, cost of 0.2%
results['Daily'] = plot_strategy_results('', [10,101,5], 0.002, True)

# Weekly timeframe, 10 to 100 DARWINs in steps of 5, cost of 0.2%
results['Weekly'] = plot_strategy_results('W', [10,101,5], 0.002, True)

# Monthly timeframe, 10 to 100 DARWINs in steps of 5, cost of 0.2%
results['Monthly'] = plot_strategy_results('M', [10,101,5], 0.002, True)

def generate_comparable_dataset(_strategy, _volatility, _timeframe):
    
    _voldf = _volatility[_volatility.index.dayofweek < 5]
    
    if _timeframe != '':
        if _timeframe == 'M':
            _voldf = _voldf.vol_portfolio.resample('M').last()
        elif _timeframe == 'W':
            _voldf = _voldf.vol_portfolio.resample('W-FRI').last()
    
    _retdf = _strategy[_strategy.index.isin(_voldf.index)]
    _retdf = _retdf[_retdf != 0]
    
    _vol = _voldf[_voldf.index.isin(_retdf.index)] 
    
    _vol.index = _vol.index.date
    _retdf.index = _retdf.index.date
    
    if _timeframe == '':
        _vol = _vol.vol_portfolio
        
    return _retdf, _vol

def compare_strategy_to_market_volatility(_strategy, _volatility, _timeframe='M'):

    _returns, _volatility = generate_comparable_dataset(_strategy, _volatility, _timeframe)

    print(f'Correlation: {np.corrcoef(_returns, _volatility)[0][1]}')
    
    print('\nGenerating plot.. please wait..')
    _graphics._plotly_dataframe_scatter_(
                            _df=pd.DataFrame(data={'strategy': zscore(_returns.values),
                                                   'volatility': zscore(_volatility.values)},
                                             index=_volatility.index),
                            _x_title = 'Date / Time',
                            _y_title = 'Z Score',
                            _main_title = f'Strategy Returns vs FX Market Volatility',
                            _plot_only = True)
    
def plot_all_test_correlations_to_market_volatility(_data,
                                                    _volatility,
                                                    _timeframe='M'):
    
    _tests = len(_data[1])
    _corrs = {}
    
    for i in range(_tests):
        _rets, _vol = generate_comparable_dataset(_data[1][i][0], _volatility, _timeframe)
        _corrs[i] = np.corrcoef(_rets, _vol)[0][1]
        
    print('\nGenerating plot.. please wait..')
    _graphics._plotly_dataframe_scatter_(
                            _df=pd.DataFrame(data={'correlation': _corrs},
                                             index=range(len(_corrs))),
                            _x_title = 'Date / Time',
                            _y_title = 'Correlation',
                            _main_title = f'{_timeframe} - Test Correlations with FX Market Volatility',
                            _plot_only = True)


    plot_all_test_correlations_to_market_volatility(results['Daily'], voldf, '')

plot_all_test_correlations_to_market_volatility(results['Weekly'], voldf, 'W')

plot_all_test_correlations_to_market_volatility(results['Monthly'], voldf, 'M')

# Compare ('', 100, 0.002) returns to FX market volatility
compare_strategy_to_market_volatility(results['Daily'][1][9][0], voldf, '')

# Compare ('W', 100, 0.002) returns to FX market volatility
compare_strategy_to_market_volatility(results['Weekly'][1][9][0], voldf, 'W')

# Compare ('M', 100, 0.002) returns to FX market volatility
compare_strategy_to_market_volatility(results['Monthly'][1][9][0], voldf, 'M')

