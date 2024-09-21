import datetime
import os
from copy import deepcopy
from functools import cache
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

import warnings

warnings.filterwarnings("ignore")

LOCAL_PATH = ""


@cache
def getAllStockPrices(stockName):
    START_DATE = datetime.datetime.now() + datetime.timedelta(days=-100000)
    END_DATE = datetime.datetime.now() + datetime.timedelta(days=1)
    ticker = yf.Ticker(stockName)
    data = ticker.history(end=END_DATE, start=START_DATE)
    data.index = data.index.date
    return data


@cache
def getStockPriceForDate(stockName, date):
    data = getAllStockPrices(stockName=stockName)
    return data.loc[date]["Close"]


@cache
def getDividendForDate(stockName, date):
    data = getAllStockPrices(stockName=stockName)
    return data.loc[date]["Dividends"]


@cache
def getStockPrices(stockName, numDays, endDate):
    startDate = endDate + datetime.timedelta(days=-2 * numDays - 20)
    ticker = yf.Ticker(stockName)
    print(f"Getting stock prices for {stockName} from {startDate} to {endDate}")
    data = ticker.history(end=endDate, start=startDate, keepna=True)

    print(f"Got stock prices for {stockName} from {startDate} to {endDate}")
    print(f"Data is {data}")

    return data["Close"][-numDays:]


@cache
def getVolatility(stockName, numDays, endDate):
    startDate = endDate + datetime.timedelta(days=-2 * numDays)
    ticker = yf.Ticker(stockName)
    data = ticker.history(end=endDate, start=startDate)

    stockChanges = np.diff(data["Close"])[-numDays:]
    volatility = np.std(stockChanges)
    return volatility


def getInverseVolatilityPositions(
    stockNames: List[str], numDays, endDate, coeficients: Dict = None
) -> Dict[str, float]:
    volatilities = [
        getVolatility(stockName=stockName, numDays=numDays, endDate=endDate)
        for stockName in stockNames
    ]
    inverseVolatilities = [1 / i for i in volatilities]
    if coeficients is None:
        return {
            stockName: i / sum(inverseVolatilities)
            for stockName, i in zip(stockNames, inverseVolatilities)
        }
    else:
        totalPosition = sum(
            coeficients[i] * u for i, u in zip(stockNames, inverseVolatilities)
        )
        return {
            stockName: coeficients[stockName] * inverseVolatility / totalPosition
            for stockName, inverseVolatility in zip(stockNames, inverseVolatilities)
        }


@cache
def getStockTradingDays(stockName, startDate, endDate):
    ticker = yf.Ticker(stockName)
    data = ticker.history(end=endDate, start=startDate)
    return [i.date() for i in data.index]


@cache
def getAllStockTradingDays(stockName):
    START_DATE = datetime.datetime.now() + datetime.timedelta(days=-100000)
    END_DATE = datetime.datetime.now() + datetime.timedelta(days=1)
    ticker = yf.Ticker(stockName)
    data = ticker.history(end=END_DATE, start=START_DATE)
    return [i.date() for i in data.index]


def isStockTradingDay(stockName, date):
    return date in getAllStockTradingDays(stockName)


@cache
def getEconomicData(filename: str) -> pd.DataFrame:
    path = os.path.join(f"{LOCAL_PATH}/economicsData", filename)
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    data.index = pd.to_datetime(data.index).date
    data["data"] = pd.to_numeric(data["data"], errors="coerce").ffill()

    return data


@cache
def getEconomicDataDates(filename: str, startDate, endDate) -> List[datetime.datetime]:
    dates = getEconomicData(filename=filename).index.to_list()
    start, end = max(startDate, dates[0]), min(dates[-1], endDate)
    results = []
    while start <= end:
        results.append(deepcopy(start))
        start += datetime.timedelta(days=1)

    return results


@cache
def getEconomicDataForDate(date: datetime.datetime, filename: str) -> float:
    data = getEconomicData(filename=filename)
    return float(data[data.index <= date].iloc[-1])
