from typing import List

from scipy import stats

from utils import (
    getEconomicDataDates,
    getEconomicDataForDate,
    getStockPriceForDate,
    getStockTradingDays,
)


def pearsonr_ci(x, y, confidenceLevel=0.99):
    result = stats.pearsonr(x, y)
    low, high = result.confidence_interval(confidence_level=confidenceLevel)
    return result.correlation, low, high


REAL_NAMES = {
    "^GSPC": "S&P 500",
    "^GSCI": "Gold Index",
    "vixIndex": "CBOE Volatility Index",
    "fedFundsRate": "Fed Funds Rate %",
    "cpiTotal": "CPI Total %",
    "coreInflationUS": "Core Inflation % (US)",
    "^DJI": "Dow Jones Industrial Average",
    "^IXIC": "NASDAQ Composite",
    "^NYA": "NYSE COMPOSITE",
    "^RUT": "Russell 2000",
    "VBMFX": "Vanguard Total Bond Fund",
    "TLT": "Long term treasuries ETF (TLT)",
    "GLD": "Gold (GLD)",
}

STOCKS = ["^GSPC", "^DJI", "^IXIC", "^NYA", "^RUT", "VBMFX", "TLT", "GLD"]
ECONOMIC_DATA = ["vixIndex", "fedFundsRate", "cpiTotal", "coreInflationUS"]


USE_RETURNS = {"^GSPC"}

TOLERANCE = {
    "^GSPC": 0.01,
    "vixIndex": 5.0,
    "fedFundsRate": 3.0,
    "cpiTotal": 1.0,
    "coreInflationUS": 3.0,
}


def getCommonDates(startDate, endDate, instruments):
    tradingDays = []
    for inst in instruments:
        if inst in STOCKS:
            tradingDays.append(
                getStockTradingDays(
                    stockName=inst, startDate=startDate, endDate=endDate
                )
            )
        elif inst in ECONOMIC_DATA:
            tradingDays.append(
                getEconomicDataDates(
                    filename=f"{inst}.csv",
                    startDate=startDate,
                    endDate=endDate,
                )
            )
        else:
            raise ValueError(f"Invalid instrument: {inst}")

    return [
        day
        for day in tradingDays[0]
        if all(day in tradingDays[i] for i in range(1, len(tradingDays)))
    ]


def getPrices(instrumentName, dates) -> List[float]:
    if instrumentName in STOCKS:
        prices = [getStockPriceForDate(instrumentName, date) for date in dates]
    elif instrumentName in ECONOMIC_DATA:
        prices = [
            getEconomicDataForDate(date=date, filename=f"{instrumentName}.csv")
            for date in dates
        ]
    else:
        raise ValueError(f"Invalid instrument: {instrumentName}")

    return prices
