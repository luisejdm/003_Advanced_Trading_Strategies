import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#sns.set_theme()

plt.rcParams['figure.figsize'] = [14, 8]
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.fancybox'] = True
plt.rcParams['figure.dpi'] = 100


def plot_cointegrated_stocks(data: pd.DataFrame)-> None:
    """
    Plot the price series of two cointegrated stocks.
    Args:
        data (pd.DataFrame): DataFrame containing the price series of two stocks.
    """
    plt.figure()
    plt.plot(data.index, data.iloc[:, 0], label=data.columns[0], color='cadetblue')
    plt.plot(data.index, data.iloc[:, 1], label=data.columns[1], color='darkslateblue')
    plt.title('Price Series of Cointegrated Stocks')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()