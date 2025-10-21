import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d.art3d import rotate_axes

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


def plot_estimations(index: pd.Index, w_pred: list, ) -> None:
    """
    Plot the predicted spread values.
    Args:
        index (np.ndarray): Array of dates corresponding to the predictions.
        w_pred (list): List of predicted spread values.
    """
    w_df = pd.DataFrame(w_pred, columns=['alpha', 'beta'], index=index)
    # Plot alpha and beta over time with left and right different y-axes
    fig, ax1 = plt.subplots()
    ax1.set_title('Estimated α and β Over Time')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('α', rotation=0, labelpad=15)
    ax1.plot(w_df.index, w_df['alpha'], color='cadetblue', label='α')
    ax1.tick_params(axis='y')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('β', rotation=0, labelpad=15)
    ax2.plot(w_df.index, w_df['beta'], color='darkslateblue', label='β')
    ax2.tick_params(axis='y')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    plt.show()


def plot_portfolio_value(dates: pd.Index, portfolio_values: list, signals: np.ndarray) -> None:
    """
    Plot the portfolio value over time.
    Args:
        dates (pd.Index): Index of dates.
        portfolio_values (list): List of portfolio values corresponding to the dates.
        signals (np.ndarray): Array of trading signals corresponding to the dates.
    """
    # Subplots con proporción de alturas: el portafolio más grande
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 10),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    ax1.plot(dates, portfolio_values, label='Portfolio Value', color='darkslateblue')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Portfolio Value')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.plot(dates, signals, label='Trading Signal', color='darkslateblue', drawstyle='steps-post')
    ax2.set_title('Trading Signals Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Signal')
    ax2.set_ylim(-2, 2)
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_spread_and_signal(dates: pd.Index, spread: np.ndarray, signals: np.ndarray, z_threshold: float) -> None:
    """
    Plot the spread and trading signals over time in subplots.
    Args:
        dates (pd.Index): Index of dates.
        spread (np.ndarray): Array of spread values corresponding to the dates.
        signals (np.ndarray): Array of trading signals corresponding to the dates.
        z_threshold (np.ndarray): Array of z-threshold values for reference.
    """
    # El subplot superior será 3 veces más alto que el inferior
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 10),
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax1.plot(dates, spread, label='Spread', color='darkslateblue')
    ax1.axhline(y=z_threshold, color='firebrick', linestyle='--', linewidth=2.5, label=f'Z-Score Thresholds')
    ax1.axhline(y=-z_threshold, color='firebrick', linestyle='--', linewidth=2.5)
    ax1.text(dates[0], z_threshold + 0.1, f'+{z_threshold:.4f}', color='firebrick', fontsize=14, va='bottom', fontweight='bold')
    ax1.text(dates[0], -z_threshold - 0.1, f'-{z_threshold:.4f}', color='firebrick', fontsize=14, va='top', fontweight='bold')
    ax1.set_title('Spread Over Time')
    ax1.set_ylabel('Spread')
    ax1.set_ylim(spread.min()-0.5, spread.max()+0.5)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.plot(dates, signals, label='Trading Signal', color='darkslateblue', drawstyle='steps-post')
    ax2.set_title('Trading Signals Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Signal')
    ax2.set_ylim(-2, 2)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()