import dataclasses
import datetime
from math import pow
from typing import Callable, Dict, List, Tuple

from tqdm import tqdm

from dynamicPortfolioOptimisation.expectedDistributions import SPREAD
from plots.plotUtils import getPrices
from strategy import Basket, Strategy
from utils import getStockTradingDays


@dataclasses.dataclass
class SimulationResult:
    initialBalance: float
    endBalance: float
    startDate: datetime.datetime
    endDate: datetime.datetime
    balancePerDay: Dict[datetime.date, float]
    basketsPerDay: Dict[datetime.date, Basket]
    contributionsPerDay: Dict[datetime.date, float]
    strategy: Strategy | None = None

    def getCARG(self) -> float:
        return (
            pow(
                self.endBalance / self.initialBalance,
                365 / (self.endDate - self.startDate).days,
            )
            - 1
        ) * 100

    def getMaxDrawdown(self) -> float:
        maxDrawdown = 0
        dailyBalances = list(self.balancePerDay.values())
        for idx, balance in enumerate(dailyBalances):
            maxDrawdown = max(maxDrawdown, 1 - balance / max(dailyBalances[: idx + 1]))
        return maxDrawdown * 100

    def getDailyReturns(self) -> list[float]:
        return {
            date: (self.balancePerDay[date] - self.balancePerDay[previousDate])
            / self.balancePerDay[previousDate]
            for date, previousDate in zip(
                list(self.balancePerDay.keys())[1:],
                list(self.balancePerDay.keys())[:-1],
            )
        }

    def getDailyReturnsWithoutContributions(self) -> list[float]:
        return {
            date: (
                self.balancePerDay[date]
                - self.balancePerDay[previousDate]
                - self.contributionsPerDay[date]
            )
            / self.balancePerDay[previousDate]
            for date, previousDate in zip(
                list(self.balancePerDay.keys())[1:],
                list(self.balancePerDay.keys())[:-1],
            )
        }

    def getMeanDailyReturn(self) -> float:
        return sum(self.getDailyReturns().values()) / len(self.getDailyReturns())

    def getMeanDailyReturnWithoutContributions(self) -> float:
        return sum(self.getDailyReturnsWithoutContributions().values()) / len(
            self.getDailyReturnsWithoutContributions()
        )

    def getReturnsStandardDeviation(self) -> float:
        dailyReturns = self.getDailyReturns().values()
        mean = sum(dailyReturns) / len(dailyReturns)
        return (sum((x - mean) ** 2 for x in dailyReturns) / len(dailyReturns)) ** 0.5

    def getSharpeRatio(self) -> float:
        return self.getMeanDailyReturn() / self.getReturnsStandardDeviation() * 250**0.5

    def getSharpeRatioWithoutContributions(self) -> float:
        return (
            self.getMeanDailyReturnWithoutContributions()
            / self.getReturnsStandardDeviation()
            * 250**0.5
        )

    def getCARGWithoutContributions(self) -> float:
        return (
            pow(
                max(
                    self.endBalance / self.initialBalance
                    - sum(self.contributionsPerDay.values()) / self.initialBalance,
                    1e-20,
                ),
                365 / (self.endDate - self.startDate).days,
            )
            - 1
        ) * 100

    def getTotalContributions(self) -> float:
        return sum(self.contributionsPerDay.values())

    def getAverageLeverage(self) -> float:
        return sum(
            [
                basket.getLeverage(date=date)
                for date, basket in self.basketsPerDay.items()
            ]
        ) / len(self.basketsPerDay)

    def getAverageEquityExposure(self) -> float:
        return sum(
            [
                sum(basket.getEquityExposures(date=date).values())
                for date, basket in self.basketsPerDay.items()
            ]
        ) / len(self.basketsPerDay)

    def getAverageBondExposure(self) -> float:
        return sum(
            [
                sum(basket.getBondExposures(date=date).values())
                for date, basket in self.basketsPerDay.items()
            ]
        ) / len(self.basketsPerDay)

    def getAverageCommodityExposure(self) -> float:
        return sum(
            [
                sum(basket.getCommodityExposures(date=date).values())
                for date, basket in self.basketsPerDay.items()
            ]
        ) / len(self.basketsPerDay)

    def getWorstDays(
        self, n: int = 10
    ) -> List[Tuple[datetime.date, float, Tuple[float, float]]]:
        return sorted(
            [
                (date, loss, (balance, previousBalance))
                for date, loss, balance, previousBalance in zip(
                    list(self.balancePerDay.keys())[1:],
                    [
                        1 - bal / previousBal
                        for bal, previousBal in zip(
                            list(self.balancePerDay.values())[1:],
                            list(self.balancePerDay.values())[:-1],
                        )
                    ],
                    list(self.balancePerDay.values())[1:],
                    list(self.balancePerDay.values())[:-1],
                )
            ],
            key=lambda x: x[2][0] / x[2][1],
        )[:n]

    def __str__(self) -> str:
        return f"""
        Initial balance: {self.initialBalance:.2f}
        End balance: {self.endBalance:.2f}
        CARG: {self.getCARG():.2f}%
        Mean daily return: {sum(self.getDailyReturns().values()) / len(self.getDailyReturns()) * 100:.2f}%
        Return standard deviation: {self.getReturnsStandardDeviation() * 100:.2f}%
        Sharpe ratio: {self.getSharpeRatio():.2f}
        Max drawdown: {self.getMaxDrawdown():.2f}%
        Total contributions: {self.getTotalContributions():.2f}
        CARG without contributions: {self.getCARGWithoutContributions():.2f}%
        Mean daily return without contributions: {self.getMeanDailyReturnWithoutContributions() * 100:.2f}%
        Sharpe ratio without contributions: {self.getSharpeRatioWithoutContributions():.2f}
        Average leverage: {self.getAverageLeverage():.2f}
        Average equity exposure: {self.getAverageEquityExposure():.2f}
        Average bond exposure: {self.getAverageBondExposure():.2f}
        Average commodity exposure: {self.getAverageCommodityExposure():.2f}
        Worst days: {self.getWorstDays()}
        """


def simulate(
    strategy: Strategy,
    startDate,
    endDate,
    initialBalance=10000,
    dailyIncome: Callable = lambda date: 0.0,
) -> SimulationResult:
    tradingDays = [
        getStockTradingDays(stockName=inst, startDate=startDate, endDate=endDate)
        for inst in strategy.tradingDaysMustHaveInstruments
    ]
    tradingDays = [
        day
        for day in tradingDays[0]
        if all(day in tradingDays[i] for i in range(1, len(tradingDays)))
    ]

    print(
        f"Trading days: {len(tradingDays)}, start date: {tradingDays[0]}, end date: {tradingDays[-1]}"
    )

    basket = strategy.getPositionsFromBasket(
        date=tradingDays[0],
        currentBasket=Basket(cash=initialBalance, date=tradingDays[0]),
    )
    balancePerDay = {
        tradingDays[0]: basket.getCashValue(tradingDays[0]),
    }
    basketsPerDay = {
        tradingDays[0]: basket,
    }

    contributionsPerDay = {tradingDays[0]: 0}

    for date in tqdm(tradingDays[1:], desc="Simulating"):
        basket = strategy.getPositionsFromBasket(date=date, currentBasket=basket)
        basket.cash += dailyIncome(date=date)
        interestRate = getPrices("fedFundsRate", [date])[0] / 100
        basket.cash *= (
            (1 + (interestRate - SPREAD) / 250)
            if basket.cash > 0
            else (1 + (interestRate + SPREAD) / 250)
        )
        balancePerDay[date] = basket.getCashValue(date)
        basketsPerDay[date] = basket
        contributionsPerDay[date] = dailyIncome(date=date)

    return SimulationResult(
        initialBalance=initialBalance,
        endBalance=basket.getCashValue(tradingDays[-1]),
        startDate=tradingDays[0],
        endDate=tradingDays[-1],
        balancePerDay=balancePerDay,
        basketsPerDay=basketsPerDay,
        contributionsPerDay=contributionsPerDay,
        strategy=strategy,
    )
