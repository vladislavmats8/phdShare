import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from stocksInfo import INSTRUMENT_TYPES
from utils import getDividendForDate, getStockPriceForDate, isStockTradingDay

CASH: str = "Cash"


class Basket:
    def __init__(
        self,
        stockPositions: Dict = None,
        cash: float = 0,
        date: datetime.datetime | None = None,
    ):
        self.stockPositions = (
            defaultdict(int, stockPositions)
            if stockPositions is not None
            else defaultdict(int)
        )
        self.cash = cash
        self.date = date

    def getCashValue(self, date, adjustForDividends: bool = False):
        return (
            sum(
                self.getPositionValue(
                    stockName=stockName,
                    date=date,
                    adjustForDividends=adjustForDividends,
                )
                for stockName in self.stockPositions
            )
            + self.cash
        )

    def getPositionValue(self, stockName, date, adjustForDividends: bool = False):
        return self.stockPositions[stockName] * (
            getStockPriceForDate(stockName=stockName, date=date)
            + (
                getDividendForDate(stockName=stockName, date=date)
                if adjustForDividends
                else 0
            )
        )

    @classmethod
    def asLinearCombinationOfBaskets(cls, basketsWithSpis: List[Tuple]):
        resultingBasket = Basket()
        for spi, basket in basketsWithSpis:
            for stock, position in basket.stockPositions.items():
                resultingBasket.stockPositions[stock] += spi * position
            resultingBasket.cash += spi * basket.cash

        return resultingBasket

    def str(self, date: datetime.datetime):
        return f"""
        Stocks: {self.stockPositions}, 
        Cash: {self.cash},
        Stock position values: {[self.getPositionValue(stockName=stockName, date=date, adjustForDividends=True) for stockName in self.stockPositions]}
        Stock exposures: {self.getExposures(date=date)}
        """

    def getExposures(self, date: datetime.datetime):
        totalValue = self.getCashValue(date=date, adjustForDividends=True)
        exposures = {
            stockName: self.getPositionValue(
                stockName=stockName, date=date, adjustForDividends=True
            )
            / totalValue
            for stockName in self.stockPositions
        }

        exposures[CASH] = self.cash / totalValue
        return exposures

    def getLeverage(self, date: datetime.datetime):
        return 1.0 - self.cash / self.getCashValue(date=date, adjustForDividends=True)

    def getEquityExposures(self, date: datetime.datetime):
        return {
            name: exposure
            for name, exposure in self.getExposures(date=date).items()
            if INSTRUMENT_TYPES.get(name) == "Equity"
        }

    def getBondExposures(self, date: datetime.datetime):
        return {
            name: exposure
            for name, exposure in self.getExposures(date=date).items()
            if INSTRUMENT_TYPES.get(name) == "Bond"
        }

    def getCommodityExposures(self, date: datetime.datetime):
        return {
            name: exposure
            for name, exposure in self.getExposures(date=date).items()
            if INSTRUMENT_TYPES.get(name) == "Commodity"
        }


def getBasketFromStockValues(
    stockValues: Dict,
    date: datetime.datetime,
    basketValue: float,
) -> Basket:
    if CASH in stockValues:
        cash = stockValues[CASH]
        stockValues.pop(CASH)
    else:
        cash = 0

    sumOfStockValues = sum(stockValues.values()) + cash
    basket = Basket(cash=cash * (basketValue / sumOfStockValues), date=date)
    for stockName, stockValue in stockValues.items():
        basket.stockPositions[stockName] = (
            stockValue
            / getStockPriceForDate(stockName=stockName, date=date)
            * (basketValue / sumOfStockValues)
        )

    return basket


def getBasketFromStockExposures(
    stockExposures: Dict,
    date: datetime.datetime,
    basketValue: float,
    skipNotTraded: bool = False,
) -> Basket:
    basket = Basket(cash=basketValue, date=date)
    totalExporsures = sum(stockExposures.values())
    tradedExposure = 0

    for stockName, stockExposure in stockExposures.items():
        if skipNotTraded and not isStockTradingDay(stockName=stockName, date=date):
            continue

        tradedExposure += stockExposure

    if tradedExposure != totalExporsures and not skipNotTraded:
        raise ValueError(
            f"Total exposure is {totalExporsures}, traded exposure is {tradedExposure}"
        )

    if tradedExposure == 0:
        raise ValueError(
            f"Total exposure is {totalExporsures}, traded exposure is {tradedExposure}"
        )

    for stockName, stockExposure in stockExposures.items():
        if skipNotTraded and not isStockTradingDay(stockName=stockName, date=date):
            continue

        stockValue = basketValue * stockExposure * (totalExporsures / tradedExposure)
        basket.stockPositions[stockName] = stockValue / getStockPriceForDate(
            stockName=stockName, date=date
        )
        basket.cash -= stockValue

    return basket


class Strategy:
    def __init__(
        self,
        possibleInstruments: List,
        tradingDaysMustHaveInstruments: Optional[List[str]] = None,
    ):
        self.possibleInstruments = possibleInstruments
        if tradingDaysMustHaveInstruments is None:
            self.tradingDaysMustHaveInstruments = self.possibleInstruments
        else:
            self.tradingDaysMustHaveInstruments = tradingDaysMustHaveInstruments

    def getPositionsFromBasket(self, date, currentBasket) -> Basket:
        pass


class ConstantExposuresStrategy(Strategy):
    def __init__(self, stockExposures: Dict, possibleInstruments: List):
        super().__init__(possibleInstruments)
        self.stockExposures = stockExposures

    def getPositionsFromBasket(self, date, currentBasket) -> Basket:
        return getBasketFromStockExposures(
            stockExposures=self.stockExposures,
            date=date,
            basketValue=currentBasket.getCashValue(date, adjustForDividends=True),
        )
