import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


def plot_all_pairs(train: pd.DataFrame, coint_results: dict, estandarize_fn) -> None:
    """
    Recorre todos los pares cointegrados {sector: [(t1,t2,p)]},
    estandariza cada par con estandarize_fn y los grafica.
    """
    for sector_name, pairs in coint_results.items():
        for (t1, t2, pvalue) in pairs:
            pair_df = train[[t1, t2]].dropna()
            if pair_df.shape[0] < 20:
                continue
            std_df = estandarize_fn(pair_df)
            plt.figure()
            plt.plot(std_df.index, std_df.iloc[:, 0], label=std_df.columns[0])
            plt.plot(std_df.index, std_df.iloc[:, 1], label=std_df.columns[1])
            plt.title(f'{sector_name}: {t1} vs {t2}  |  p={pvalue:.4f}')
            plt.xlabel('Date')
            plt.ylabel('Price (standardized)')
            plt.legend()
            plt.grid(True)
            plt.show()


def plot_w(w_pred: list, index: np.ndarray) -> None:
    """
    Plot the predicted spread values.
    Args:
        w_pred (list): List of predicted spread values.
        index (np.ndarray): Array of dates corresponding to the predictions.
    """
    w_df = pd.DataFrame(w_pred, columns=['Intercept', 'Delta Hedge'], index=index)
    plt.figure()
    plt.plot(w_df.index, w_df.iloc[:, 0], label='Intercept')
    plt.plot(w_df.index, w_df.iloc[:, 1], label='Delta Hedge')
    plt.title('Predicted Spread Values Over Time')
    plt.xlabel('Date')
    plt.ylabel('Spread Value')
    plt.legend()
    plt.grid(True)
    plt.show()

