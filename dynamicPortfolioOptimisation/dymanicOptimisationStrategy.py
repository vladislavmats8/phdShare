from copy import deepcopy
import datetime
from typing import Callable, List, Tuple

from tqdm import tqdm

from dynamicPortfolioOptimisation.constants import NEGATIVE_STEPS, TOTAL_STEPS
from dynamicPortfolioOptimisation.distribution import (
    Distribution,
    logToNumber,
    numberToLog,
)
from dynamicPortfolioOptimisation.expectedDistributions import getExpectedDistributions
from plots.plotUtils import getCommonDates
from simulation import SimulationResult, simulate
from strategy import (
    Basket,
    ConstantExposuresStrategy,
    Strategy,
    getBasketFromStockExposures,
)


class DynamicOptimisationStrategy(Strategy):
    def __init__(
        self,
        possibleInstruments: List[str],
        daysPerRebalance: int,
        startDate: datetime.date,
        endDate: datetime.date,
        dailyIncome: Callable,
        initialBalance: float,
        utilityFunction: Callable,
        dailyUtilityFunction: Callable,
        strategyName: str,
    ):
        super().__init__(possibleInstruments)
        self.daysPerRebalance = daysPerRebalance
        self.startDate = startDate
        self.endDate = endDate
        tradingDays = getCommonDates(startDate, endDate, self.possibleInstruments)
        self.periodsToSim = [
            tradingDays[i : i + daysPerRebalance]
            for i in range(0, len(tradingDays), daysPerRebalance)
        ]
        self.strategyName = strategyName

        self.daysToPeriodIdx = {
            day: idx for idx, period in enumerate(self.periodsToSim) for day in period
        }
        self.distributionsPerPeriod = [
            getExpectedDistributions(
                datetime.date(2014, 3, 1) or self.startDate,
                self.possibleInstruments,
                periodDays=daysPerRebalance,
            )
            for period in tqdm(
                self.periodsToSim,
                desc="Getting expected distributions",
                leave=False,
                total=len(self.periodsToSim),
            )
        ]

        self.dp = {
            period: {money: None for money in range(TOTAL_STEPS)}
            for period in range(len(self.periodsToSim) + 1)
        }
        self.initialMoney = int(numberToLog(initialBalance))
        self.utilityFunction = utilityFunction
        self.dailyUtilityFunction = dailyUtilityFunction
        self.dailyIncome = dailyIncome
        self.incomePerPeriod = [
            sum(dailyIncome(day) for day in period) for period in self.periodsToSim
        ]
        self.optimalInitialDistribution = self.getOptimalSolution()

    def getPositionsFromBasket(self, date, currentBasket) -> Basket:
        cashValue = currentBasket.getCashValue(date, adjustForDividends=True)
        logMoney = max(round(numberToLog(cashValue)), 0)
        periodIdx = self.daysToPeriodIdx[date]

        finalDist, optimalPortfolio, _, _ = self.dp[periodIdx][logMoney]
        optimalPortfolio = deepcopy(optimalPortfolio)

        INDEX_TO_INSTRUMENT = {
            "^IXIC": "QQQ",
            "^GSPC": "SPY",
            "^DJI": "DIA",
            "^NYA": "VTI",
            "^RUT": "IWM",
            "VBMFX": "BND",
        }

        optimalPortfolio.portfolio = {
            INDEX_TO_INSTRUMENT.get(instrument, instrument): exposure
            for instrument, exposure in optimalPortfolio.portfolio.items()
        }
        basket = getBasketFromStockExposures(
            stockExposures=optimalPortfolio.portfolio,
            date=date,
            basketValue=cashValue,
        )
        return basket

    def computeSolutionUtilityOnly(
        self,
        period: int,
        minMoney: int,
        maxMoney: int,
        lowerBoundDistIdx: int,
        upperBoundDistIdx: int,
    ):
        money = (minMoney + maxMoney) // 2
        income = self.incomePerPeriod[period]
        bestUtility = None
        bestPortfolioIdx = 0
        retusnDistributions = self.distributionsPerPeriod[period]
        dates = self.periodsToSim[period]
        for idxDist, dist in enumerate(retusnDistributions):
            if idxDist < lowerBoundDistIdx or idxDist > upperBoundDistIdx:
                continue
            finalUtility = 0.0

            for idx, probability in dist.getSignificantProbabilities():
                moneyAfterMove = money + idx - NEGATIVE_STEPS
                moneyAfterIncome = round(
                    numberToLog(logToNumber(moneyAfterMove) + income)
                )

                moneyAfterIncome = max(moneyAfterIncome, 0.0)

                if moneyAfterIncome < 0:
                    finalUtility += probability * self.utilityFunction(moneyAfterIncome)
                    continue
                else:
                    moneyAfterIncome = min(moneyAfterIncome, TOTAL_STEPS - 1)
                    finalUtility += probability * (
                        self.dp[period + 1][moneyAfterIncome][0]
                        + self.dailyUtilityFunction(
                            x=money, y=moneyAfterIncome, date=dates[0]
                        )
                    )
            if bestUtility is None or finalUtility > bestUtility:
                bestUtility = finalUtility
                bestPortfolio = dist
                bestPortfolioIdx = idxDist

        self.dp[period][money] = bestUtility, bestPortfolio, bestPortfolioIdx, income
        if minMoney < money:
            self.computeSolutionUtilityOnly(
                period=period,
                minMoney=minMoney,
                maxMoney=money,
                lowerBoundDistIdx=bestPortfolioIdx,
                upperBoundDistIdx=upperBoundDistIdx,
            )
        if maxMoney > money + 1:
            self.computeSolutionUtilityOnly(
                period=period,
                minMoney=money + 1,
                maxMoney=maxMoney,
                lowerBoundDistIdx=lowerBoundDistIdx,
                upperBoundDistIdx=bestPortfolioIdx,
            )

    def getOptimalSolution(self) -> Tuple[Distribution, Distribution, int]:
        for money in range(TOTAL_STEPS):
            self.dp[len(self.periodsToSim)][money] = (
                self.utilityFunction(money),
                None,
                None,
                0.0,
            )

        for period in tqdm(
            range(len(self.periodsToSim) - 1, -1, -1),
            desc="Computing portfolio states",
            total=len(self.periodsToSim),
        ):
            self.computeSolutionUtilityOnly(
                period=period,
                minMoney=0,
                maxMoney=TOTAL_STEPS,
                lowerBoundDistIdx=0,
                upperBoundDistIdx=len(self.distributionsPerPeriod[period]) - 1,
            )

        return self.dp[0][self.initialMoney]


def getDefaultStrategiesResults(
    startDate: datetime.date,
    endDate: datetime.date,
    initialAmount: float,
    dailyIncome: Callable,
) -> List[SimulationResult]:
    results = []
    for exposures in [
        {
            "SPY": 1.0,
        },
        {
            "SPY": 1.5,
        },
        {
            "SPY": 2.0,
        },
        {
            "SPY": 0.5,
            "TLT": 0.5,
        },
        {
            "SPY": 0.5,
            "TLT": 0.3,
            "GLD": 0.2,
        },
        {
            "SPY": 0.5,
            "TLT": 0.5,
            "GLD": 0.5,
        },
    ]:
        constantExposuresStrategy = ConstantExposuresStrategy(
            stockExposures=exposures,
            possibleInstruments=["SPY", "TLT", "GLD"],
        )

        constantExposureResult = simulate(
            strategy=constantExposuresStrategy,
            startDate=startDate,
            endDate=endDate,
            initialBalance=initialAmount,
            dailyIncome=dailyIncome,
        )
        constantExposureResult.strategyName = ", ".join(
            f"{instrument}: {exposure:.2f}"
            for instrument, exposure in exposures.items()
        )
        results.append(constantExposureResult)

    return results


def getMatchingStrategyResult(
    startDate: datetime.date,
    endDate: datetime.date,
    initialAmount: float,
    dailyIncome: Callable,
    result: SimulationResult,
) -> SimulationResult:
    exposures = {
        "SPY": result.getAverageEquityExposure(),
        "TLT": result.getAverageBondExposure(),
        "GLD": result.getAverageCommodityExposure(),
    }
    constantExposuresStrategy = ConstantExposuresStrategy(
        stockExposures=exposures,
        possibleInstruments=["SPY", "TLT", "GLD"],
    )

    constantExposureResult = simulate(
        strategy=constantExposuresStrategy,
        startDate=startDate,
        endDate=endDate,
        initialBalance=initialAmount,
        dailyIncome=dailyIncome,
    )
    constantExposureResult.strategyName = f"""Matching strategy, CARG: {constantExposureResult.getCARGWithoutContributions():.2f}%
Sharpe ratio: {constantExposureResult.getSharpeRatioWithoutContributions():.2f}, Max drawdown: {constantExposureResult.getMaxDrawdown():.2f}%"""
    return constantExposureResult
