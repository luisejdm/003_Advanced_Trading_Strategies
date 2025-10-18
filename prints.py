def print_best_pair(best_pair: tuple, best_pvalue: float, sector: str) -> None:
    """
    Print the best cointegrated pair and its p-value.
    Args:
        best_pair (tuple): The best cointegrated pair.
        best_pvalue (float): The p-value of the best cointegrated pair.
        sector (str): The sector of the best cointegrated pair.
    """
    print(
        f'{'-' * 50}' +
        f'\nBest cointegrated pair: {best_pair[0]} - {best_pair[1]} ' +
        f'\nP-value: {best_pvalue:.6f}' +
        f'\nSector: {sector}'
    )