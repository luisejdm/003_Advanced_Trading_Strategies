def get_sectors() -> dict:
    """
    Returns lists of stock tickers categorized by sectors: Tech, Comm, Consum, Health, Finan.
    Returns:
        dict: A dictionary with sector names as keys and lists of stock tickers as values.
    """
    sectors_dict = {
        'Information Technology': [
        "MSFT", "AAPL", "INTC", "CSCO", "QCOM",
        "ORCL", "TXN", "ADBE", "HPQ", "AVGO"
    ],

        'Communication Services': [
        "EA", "GOOG", "IPG", "VZ", "T",
        "DIS", "NFLX", "CMCSA", "CHTR", "OMC"
    ],

        'Consumer Discretionary': [
    'AMZN', 'MCD', 'HD', 'NKE', 'SBUX',
    'LOW', 'TJX', 'TGT', 'MAR', 'ROST'
    ],

        'Health Care': [
        'AMZN', 'MCD', 'HD', 'NKE', 'SBUX',
        'LOW', 'TJX', 'TGT', 'MAR', 'ROST'
    ],

        'Financials': [
        'JPM', 'BAC', 'WFC', 'C', 'GS',
        'MS', 'BK', 'AXP', 'SPGI', 'BLK'
    ]

    }
    return sectors_dict