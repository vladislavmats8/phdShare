import datetime
import itertools
from multiprocessing import Pool
import os
from typing import Callable

from dynamicPortfolioOptimisation.visualization import (
    visualizeAssetClassExposuresDifferentPlots,
    visualizeOptimalPortfolioLeverage,
    visualizeStateProbabilities,
    visualizeUtilityValues,
)
import numpy as np
from dynamicPortfolioOptimisation.constants import LOG_BASE, TOTAL_STEPS
from dynamicPortfolioOptimisation.distribution import logToNumber, numberToLog
from dynamicPortfolioOptimisation.dymanicOptimisationStrategy import (
    DynamicOptimisationStrategy,
    getDefaultStrategiesResults,
    getMatchingStrategyResult,
)
from simulation import SimulationResult, simulate


os.environ["TQDM_DISABLE"] = "1"

import warnings

warnings.filterwarnings("ignore")


def dailyIncome_no_income(date):
    return 0.0


def dailyIncome_linearly_increasing(date, startDate):
    return (date - startDate).days / 365 / 250 * 1e4


def dailyIncome_1e4_per_year(date, startDate):
    return 1e4 / 250


daily_income_functions = {
    "no_income": lambda startDate: dailyIncome_no_income,
    "linearly_increasing": lambda startDate: lambda date: dailyIncome_linearly_increasing(
        date, startDate
    ),
    "1e4_per_year": lambda startDate: lambda date: dailyIncome_1e4_per_year(
        date, startDate
    ),
}

utility_functions = {}

utility_functions.update(
    {
        "Final utility log": lambda x: x / TOTAL_STEPS if x > 0 else -1e-9,
        **{
            f"Final utility inverse log pow {i}": lambda x, i=i: (
                -((1 - x / TOTAL_STEPS) ** i) + 1 if x > 0 else -1
            )
            for i in np.linspace(1, 15, 15)
        },
    }
)

inverse_utility_functions = {}

inverse_utility_functions.update(
    {
        "Final utility log": lambda x: logToNumber(x * TOTAL_STEPS),
        **{
            f"Final utility inverse log pow {i}": lambda x, i=i: logToNumber(
                TOTAL_STEPS * (1 - ((1 - x) ** (1 / i)))
            )
            for i in np.linspace(1, 15, 15)
        },
    }
)


def getPathUtility(
    result: SimulationResult, utilityFunction: Callable, dailyUtilityFunction: Callable
) -> float:
    totalUtility = 0.0

    for date, balance, previousBalance in zip(
        result.balancePerDay.keys(),
        result.balancePerDay.values(),
        list(result.balancePerDay.values())[1:],
    ):
        totalUtility += dailyUtilityFunction(
            numberToLog(balance), numberToLog(previousBalance), date
        )

    totalUtility += utilityFunction(numberToLog(result.endBalance))

    return totalUtility


def run_simulation(
    possibleInstruments,
    daysPerRebalance,
    startDate,
    endDate,
    initialBalance,
    dailyIncomeName,
    utilityFunctionName,
    lossAversion,
    finalUtilityProjectionPower,
):
    dailyIncome = daily_income_functions[dailyIncomeName](startDate)
    utilityFunction, inverseUtilityFunction = (
        utility_functions[utilityFunctionName],
        inverse_utility_functions[utilityFunctionName],
    )

    dailyUtilityName = (
        f"Daily utility {-lossAversion:.6f} * "
        f"(((date - startDate).days div (endDate - startDate).days) ** {finalUtilityProjectionPower})"
    )
    dailyUtilityFunction = (
        lambda x, y, date, lossAversion=lossAversion, utilityFunction=utilityFunction, finalUtilityProjectionPower=finalUtilityProjectionPower: (
            -lossAversion * (max((x - y) * np.log(LOG_BASE), 0.0) ** 2)
            + utilityFunction(x)
        )
        * ((date - startDate).days / (endDate - startDate).days)
        ** finalUtilityProjectionPower
    )

    strategyName = f"{utilityFunctionName}/{dailyUtilityName}/{dailyIncomeName} {daysPerRebalance} days per rebalance, {startDate} to {endDate}"

    if os.path.exists(f"results/optimalAssetClassExposures/{strategyName}.png"):
        print(f"Skipping {strategyName}")
        return

    os.makedirs(
        f"results/optimalAssetClassExposures/{utilityFunctionName}/{dailyUtilityName}",
        exist_ok=True,
    )
    os.makedirs(
        f"results/optimalPortfolioLeverage/{utilityFunctionName}/{dailyUtilityName}",
        exist_ok=True,
    )
    os.makedirs(
        f"results/transitions/{utilityFunctionName}/{dailyUtilityName}", exist_ok=True
    )
    os.makedirs(
        f"results/utilityValues/{utilityFunctionName}/{dailyUtilityName}", exist_ok=True
    )

    strategy = DynamicOptimisationStrategy(
        possibleInstruments=possibleInstruments,
        daysPerRebalance=daysPerRebalance,
        startDate=startDate,
        endDate=endDate,
        initialBalance=initialBalance,
        dailyIncome=dailyIncome,
        utilityFunction=utilityFunction,
        dailyUtilityFunction=dailyUtilityFunction,
        strategyName=strategyName,
    )
    result = simulate(strategy, startDate, endDate, initialBalance, dailyIncome)

    defaultStrategiesResults = getDefaultStrategiesResults(
        startDate, endDate, initialBalance, dailyIncome
    )

    matchingStrategyResult = getMatchingStrategyResult(
        startDate=startDate,
        endDate=endDate,
        initialAmount=initialBalance,
        dailyIncome=dailyIncome,
        result=result,
    )

    print(
        f"{utilityFunctionName.split(' ')[-1]}, {dailyIncomeName}, {finalUtilityProjectionPower}, U: {getPathUtility(result, utilityFunction, dailyUtilityFunction)}, MU: {getPathUtility(matchingStrategyResult, utilityFunction, dailyUtilityFunction)}, {(endDate - startDate).days / 360}, LA: {lossAversion}"
    )
    visualizeStateProbabilities(result)
    visualizeStateProbabilities(
        result, strategiesToCompareAgainst=defaultStrategiesResults
    )
    visualizeStateProbabilities(
        result, strategiesToCompareAgainst=[matchingStrategyResult]
    )
    visualizeUtilityValues(result)
    visualizeUtilityValues(result, inverseUtilityFunction=inverseUtilityFunction)
    visualizeOptimalPortfolioLeverage(result)
    visualizeAssetClassExposuresDifferentPlots(result)
    visualizeAssetClassExposuresDifferentPlots(result, normalize=True)


if __name__ == "__main__":
    np.random.seed(0)  # For reproducibility
    initialBalance = 1e4
    daysPerRebalance = 60
    startDate = datetime.date(2005, 3, 1)
    endDate = datetime.date(2006, 3, 1)
    possibleInstruments = ["^GSPC"]
    possibleInstruments = ["^GSPC", "TLT", "GLD"]

    simulation_params = []

    for (
        dailyIncomeName,
        utilityFunctionName,
        lossAversion,
        finalUtilityProjectionPower,
        datesOffset,
        additionalDays,
    ) in itertools.product(
        daily_income_functions.keys(),
        utility_functions.keys(),
        [0] + list(np.geomspace(1e-1, 1, 3)),
        [0],
        list(range(0, 360 * 20, int(360 * 1))),
        list(range(0, 360 * 0 + 1, int(360 * 1))),
    ):
        if endDate + datetime.timedelta(
            days=datesOffset + additionalDays
        ) >= datetime.date(2024, 1, 1):
            continue

        if additionalDays and datesOffset % additionalDays != 0:
            continue

        simulation_params.append(
            (
                possibleInstruments,
                daysPerRebalance,
                startDate + datetime.timedelta(days=datesOffset),
                endDate + datetime.timedelta(days=datesOffset + additionalDays),
                initialBalance,
                dailyIncomeName,
                utilityFunctionName,
                lossAversion,
                finalUtilityProjectionPower,
            )
        )

    print(f"Running {len(simulation_params)} simulations")

    with Pool(processes=16) as pool:
        pool.starmap(run_simulation, simulation_params)
