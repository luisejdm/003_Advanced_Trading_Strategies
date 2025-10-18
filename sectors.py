def get_sectors() -> dict:
    """
    Returns lists of stock tickers categorized by sectors: Tech, Comm, Consum, Health, Finan.
    Returns:
        dict: A dictionary with sector names as keys and lists of stock tickers as values.
    """
    sectors_dict = {

        'Information Technology': [
        "MSFT", "AAPL", "INTC", "CSCO", "QCOM",
        "ORCL", "TXN", "HPQ", "AVGO",
        "IBM", "ADBE", "NVDA", "AMD", "INTU",
        "MU", "STX", "ASML", "ACN"
    ],

        'Communication Services': [
        "EA", "GOOG", "IPG", "VZ", "T",
        "DIS", "NFLX", "CMCSA", "CHTR", "OMC",
        "WPP", "PUBGY", "VOD", "BCE", "TEF",
        "NYT"
    ],

        'Consumer Discretionary': [
        'AMZN', 'MCD', 'HD', 'NKE', 'SBUX',
        'LOW', 'TJX', 'TGT', 'MAR', 'ROST',
        'BBY', 'EBAY', 'BKNG', 'EXPE', 'ORLY',
        'AZO', 'AAP', 'F', 'TM',
        'HMC', 'WSM', 'KMX', 'ULTA', 'HOG'
    ],

        'Health Care': [
        "JNJ", "PFE", "MRK", "ABT", "MDT",
        "DHR", "BAX", "BDX", "LLY", "CVS",
        "AMGN", "GILD", "UNH", "HUM", "CI",
        "SYK", "TMO", "BMY", "GSK", "AZN",
        "SNY", "MCK", "CAH", "ISRG"
    ],

        'Financials': [
        'JPM', 'BAC', 'WFC', 'C', 'GS',
        'MS', 'BK', 'AXP', 'SPGI', 'BLK',
        'USB', 'PNC', 'SCHW', 'ICE', 'CME',
        'AIG', 'PRU', 'MET', 'V', 'MA',
        'COF', 'MMC', 'MSCI', 'MCO',

    ]

    }
    return sectors_dict