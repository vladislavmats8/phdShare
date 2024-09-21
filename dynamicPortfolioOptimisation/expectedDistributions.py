import datetime
from functools import lru_cache
from typing import Dict, List, Tuple


from dynamicPortfolioOptimisation.constants import MAX_LEVERAGE
from dynamicPortfolioOptimisation.cpEffecientFrontier import find_efficient_frontier
from dynamicPortfolioOptimisation.distribution import (
    Distribution,
    getLognormDistribution,
)
from plots.plotUtils import getPrices
from strategy import CASH
from utils import getAllStockPrices

SPREAD = 0.015


@lru_cache
def getMeansStdsCorrelations(
    date: datetime.date,
    instrumentsToUse: Tuple[str],
    periodDays: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
    interestRate = (getPrices("fedFundsRate", [date])[0] / 100 + SPREAD) / 250

    prices, returns = {}, {}

    for instrument in instrumentsToUse:
        prices[instrument] = getAllStockPrices(instrument).loc[:date][-1500:]
        returns[instrument] = prices[instrument]["Close"].pct_change().dropna()

    means = {
        instrument: (1 + returns[instrument].mean() - interestRate) ** periodDays - 1
        for instrument in instrumentsToUse
    }

    stds = {
        instrument: returns[instrument].std() * (periodDays**0.5)
        for instrument in instrumentsToUse
    }

    correlations = {}
    for instrument1 in instrumentsToUse:
        correlations[instrument1] = {}
        for instrument2 in instrumentsToUse:
            commonDates = sorted(
                set(returns[instrument1].index) & set(returns[instrument2].index)
            )
            returns1 = returns[instrument1].loc[commonDates]
            returns2 = returns[instrument2].loc[commonDates]
            correlations[instrument1][instrument2] = returns1.corr(returns2)

    means[CASH] = 0.0
    stds[CASH] = 0.0
    correlations[CASH] = {instrument: 0.0 for instrument in instrumentsToUse}
    for instrument in instrumentsToUse:
        correlations[instrument][CASH] = 0.0

    return means, stds, correlations


def getExpectedDistributions(
    date: datetime.date,
    instrumentsToUse: List[str],
    periodDays: int,
) -> List[Distribution]:
    means, stds, correlations = getMeansStdsCorrelations(
        date, tuple(instrumentsToUse), periodDays=periodDays
    )
    distributions = []

    for weights, portfolioReturn, portfolioStd in find_efficient_frontier(
        means, stds, correlations
    ):
        newDistribution = getLognormDistribution(portfolioReturn, portfolioStd)
        newDistribution.portfolio = {
            instrument: round(weight, 5)
            for instrument, weight in zip(means, weights)
            if abs(weight) > 1e-5 and instrument != CASH
        }
        assert all(
            abs(weight) <= MAX_LEVERAGE + 1e-4 for weight in weights
        ), f"weights: {weights}"
        assert (
            newDistribution.probabilities[0] < 1e-5
        ), f"newDistribution: {newDistribution}"
        distributions.append(newDistribution)

    return distributions
