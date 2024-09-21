import os
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

from dynamicPortfolioOptimisation.constants import (
    MAX_LEVERAGE,
    MIN_VISUALIZE_STEPS,
    NEGATIVE_STEPS,
    PROBABILITY_THR,
    TOTAL_STEPS,
)
from dynamicPortfolioOptimisation.distribution import (
    logToNumber,
    logToNumberArray,
    numberToLog,
)
from simulation import SimulationResult


def interpolateMatrix(
    arr: np.ndarray, newShapeX: int | None = None, newShapeY: int | None = None
) -> np.ndarray:
    newShapeX = newShapeX or arr.shape[1]
    newShapeY = newShapeY or arr.shape[0]
    zoom_factor_y = newShapeY / arr.shape[0]
    zoom_factor_x = newShapeX / arr.shape[1]
    return zoom(arr, (zoom_factor_y, zoom_factor_x), order=1)


def getAlmostZeroRange(arr: np.ndarray, thr: float = 1e-2) -> np.ndarray:
    nonZeroRows = []
    for i in range(arr.shape[0]):
        if i < 2 * NEGATIVE_STEPS or not np.all(
            np.isclose(arr[i, :], arr[i, arr.shape[1] - 1], atol=thr)
        ):
            nonZeroRows.append(i)
        else:
            break

    return nonZeroRows


def visualizeStateProbabilities(
    simulationResult: SimulationResult,
    strategiesToCompareAgainst: List[SimulationResult] = None,
):
    arr = calculateStateProbabilitiesMatrix(simulationResult)
    dates = list(simulationResult.balancePerDay.keys())

    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(18, 12))
    arr /= np.max(arr, axis=1).reshape(-1, 1)
    arr = arr.clip(0.0, 1.0)
    arr = np.transpose(arr)
    zeros = getAlmostZeroRange(arr)
    simulationResult.interestingRange = max(zeros[-1], 700)
    arr = arr[MIN_VISUALIZE_STEPS : simulationResult.interestingRange, :]
    caxes = ax.imshow(
        arr,
        interpolation="bicubic",
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(caxes, location="bottom", format="%.2f", pad=0.13).ax.set_xlabel(
        "Implied state probability density",
        fontdict={"size": 22},
        labelpad=13,
    )
    cumSum = np.cumsum(arr, axis=0)
    for quantile in [0.1, 0.25, 0.5, 0.75, 0.9]:
        quantiles = np.argmin(
            np.where(cumSum > quantile * np.sum(arr, axis=0), cumSum, 1e18), axis=0
        )
        ax.plot(quantiles, color="red", linestyle="--", linewidth=1)
        ax.text(len(dates), quantiles[-1], f"p{quantile * 100:.0f}%")

    if simulationResult is not None:
        ax.plot(
            [
                int(numberToLog(amount)) - MIN_VISUALIZE_STEPS
                for amount in simulationResult.balancePerDay.values()
            ],
            color="Cyan",
            linestyle="--",
            linewidth=3,
            label="Realised strategy performance",
        )

    if strategiesToCompareAgainst is not None:
        for idx, strategy in enumerate(strategiesToCompareAgainst):
            ax.plot(
                [
                    int(numberToLog(amount)) - MIN_VISUALIZE_STEPS
                    for amount in strategy.balancePerDay.values()
                ],
                linestyle="dashdot",
                linewidth=2,
                label=strategy.strategyName,
            )

    ax.set_xticks(range(len(dates)), dates, rotation=0)
    ax.xaxis.set_ticks_position("bottom")
    money = list(
        map(
            lambda x: f"${logToNumber(x) / 1e3:.1f}k".format(x),
            list(range(MIN_VISUALIZE_STEPS, MIN_VISUALIZE_STEPS + arr.shape[0])),
        )
    )
    ax.set_yticks(range(len(money)), money)
    ax.locator_params(axis="y", nbins=12)
    ax.locator_params(axis="x", nbins=6)
    ax.tick_params(axis="both", which="major", pad=15)
    ax.set_xlabel("Date", fontdict={"size": 22}, labelpad=13)
    ax.set_ylabel("Net Worth", fontdict={"size": 22}, labelpad=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fancybox=True,
        shadow=True,
        fontsize=20,
    )
    plt.title(
        "Implied state probabilities vs realised strategy performance over time\n"
        + f"CARG: {simulationResult.getCARGWithoutContributions():.2f}%, Sharpe: {simulationResult.getSharpeRatioWithoutContributions():.2f}, "
        + f"Max drawdown: {simulationResult.getMaxDrawdown():.2f}%\n"
        + f"Initial balance: ${simulationResult.initialBalance / 1e3:.1f}k, End balance: ${simulationResult.endBalance / 1e3:.1f}k, total contributions: ${sum(simulationResult.contributionsPerDay.values()) / 1e3:.1f}k"
        + f"\nAverage leverage: {simulationResult.getAverageLeverage():.2f}"
        + f", Equity exposure: {simulationResult.getAverageEquityExposure():.2f}, Bond exposure: {simulationResult.getAverageBondExposure():.2f}"
        + f", Commodity exposure: {simulationResult.getAverageCommodityExposure():.2f}",
        fontdict={"size": 22},
        pad=13,
    )
    plt.grid(True)
    fig.tight_layout()
    os.makedirs("results/transitions", exist_ok=True)
    plt.savefig(
        f"results/transitions/{simulationResult.strategy.strategyName} {len(strategiesToCompareAgainst) if strategiesToCompareAgainst else 'None'}.png"
    )


def calculateStateProbabilitiesMatrix(
    simulationResult: SimulationResult,
) -> np.ndarray:
    dp = simulationResult.strategy.dp
    numberOfDays = len(simulationResult.strategy.periodsToSim)
    transitionProbabilities = np.zeros((numberOfDays + 1, TOTAL_STEPS))
    transitionProbabilities[0, int(numberToLog(simulationResult.initialBalance))] = 1.0

    for day in tqdm(range(numberOfDays), desc="Calculating transition probabilities"):
        for money in range(TOTAL_STEPS):
            if dp[day][money] is None:
                continue

            if transitionProbabilities[day, money] < PROBABILITY_THR:
                continue

            _, distribution, idx, income = dp[day][money]

            for idx, prob in distribution.getSignificantProbabilities():
                if prob * transitionProbabilities[day, money] < 1e-10:
                    continue

                afterIncome = numberToLog(
                    logToNumber(idx + money - NEGATIVE_STEPS) + income
                )
                if afterIncome < 0 or afterIncome >= TOTAL_STEPS - 2:
                    continue

                upProb = (
                    prob
                    * (afterIncome - int(afterIncome))
                    * transitionProbabilities[day, money]
                )
                downProb = (
                    prob
                    * (1 - (afterIncome - int(afterIncome)))
                    * transitionProbabilities[day, money]
                )
                transitionProbabilities[day + 1, int(afterIncome) + 1] += upProb
                transitionProbabilities[day + 1, int(afterIncome)] += downProb

    transitionProbabilities = interpolateMatrix(
        transitionProbabilities,
        newShapeY=simulationResult.strategy.daysPerRebalance
        * transitionProbabilities.shape[0],
    )[: len(simulationResult.balancePerDay), :]

    return transitionProbabilities


def calculateUtilityValuesMatrix(simulationResult: SimulationResult) -> np.ndarray:
    dp = simulationResult.strategy.dp
    numberOfDays = len(simulationResult.strategy.periodsToSim)
    utilityValuesMatrix = np.zeros((numberOfDays, TOTAL_STEPS))
    for day in tqdm(range(numberOfDays), desc="Calculating utility values"):
        for money in range(TOTAL_STEPS):
            if dp[day][money] is None:
                continue
            utilityValuesMatrix[day, money] = dp[day][money][0]

    utilityValuesMatrix = interpolateMatrix(
        utilityValuesMatrix,
        newShapeY=simulationResult.strategy.daysPerRebalance
        * utilityValuesMatrix.shape[0],
    )[: len(simulationResult.balancePerDay), :]

    return utilityValuesMatrix


def visualizeUtilityValues(
    simulationResult: SimulationResult,
    inverseUtilityFunction: Callable | None = None,
):
    utilityValuesMatrix = calculateUtilityValuesMatrix(simulationResult).transpose()
    utilityValuesMatrix = utilityValuesMatrix[MIN_VISUALIZE_STEPS:800, :]
    if inverseUtilityFunction is None:
        utilityValuesMatrix = (utilityValuesMatrix - np.min(utilityValuesMatrix)) / (
            np.max(utilityValuesMatrix) - np.min(utilityValuesMatrix)
        )
    else:
        utilityValuesMatrix = inverseUtilityFunction(
            utilityValuesMatrix
        ) / logToNumberArray(range(MIN_VISUALIZE_STEPS, 800)).reshape(-1, 1)
    dates = list(simulationResult.balancePerDay.keys())

    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(18, 12))
    caxes = ax.matshow(
        utilityValuesMatrix,
        interpolation="bicubic",
        aspect="auto",
        origin="lower",
        cmap="cool" if inverseUtilityFunction is None else "spring",
    )
    fig.colorbar(caxes, location="top", format="{x:1.2f}", pad=0.05).ax.set_xlabel(
        (
            "State utility value"
            if inverseUtilityFunction is None
            else "Utility-implied total return"
        ),
        fontdict={"size": 22},
        labelpad=13,
    )
    ax.set_xticks(range(len(dates)), dates, rotation=0)
    ax.xaxis.set_ticks_position("bottom")
    money = list(
        map(
            lambda x: f"${logToNumber(x) / 1e3:.1f}k".format(x),
            list(
                range(
                    MIN_VISUALIZE_STEPS,
                    MIN_VISUALIZE_STEPS + utilityValuesMatrix.shape[0],
                )
            ),
        )
    )
    ax.set_yticks(range(len(money)), money)
    ax.set_xlabel("Date", fontdict={"size": 22}, labelpad=13)
    ax.set_ylabel("Net Worth", fontdict={"size": 22}, labelpad=0)
    ax.locator_params(axis="y", nbins=12)
    ax.locator_params(axis="x", nbins=6)
    ax.tick_params(axis="both", which="major", pad=15)
    plt.grid(True)
    fig.tight_layout()
    os.makedirs("results/utilityValues", exist_ok=True)
    plt.savefig(
        f"results/utilityValues/{simulationResult.strategy.strategyName}{inverseUtilityFunction is not None}.png"
    )


def calculateOptimalPortfolioLeverageMatrix(
    simulationResult: SimulationResult,
) -> np.ndarray:
    dp = simulationResult.strategy.dp
    numberOfDays = len(simulationResult.strategy.periodsToSim)
    optimalPortfolioLeverageMatrix = np.zeros((numberOfDays, TOTAL_STEPS))
    for day in tqdm(range(numberOfDays), desc="Calculating optimal portfolio leverage"):
        for money in range(TOTAL_STEPS):
            if dp[day][money] is None:
                continue
            utility, bestPortfolio, idx, income = dp[day][money]
            optimalPortfolioLeverageMatrix[day, money] = sum(
                bestPortfolio.portfolio.values()
            )

    optimalPortfolioLeverageMatrix = interpolateMatrix(
        optimalPortfolioLeverageMatrix,
        newShapeY=simulationResult.strategy.daysPerRebalance
        * optimalPortfolioLeverageMatrix.shape[0],
    )[: len(simulationResult.balancePerDay), :]
    return optimalPortfolioLeverageMatrix


def visualizeOptimalPortfolioLeverage(simulationResult: SimulationResult):
    optimalPortfolioLeverageMatrix = calculateOptimalPortfolioLeverageMatrix(
        simulationResult
    ).transpose()
    optimalPortfolioLeverageMatrix = optimalPortfolioLeverageMatrix[
        MIN_VISUALIZE_STEPS:800, :
    ]
    dates = list(simulationResult.balancePerDay.keys())

    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(18, 12))
    caxes = ax.matshow(
        optimalPortfolioLeverageMatrix,
        interpolation="bicubic",
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=MAX_LEVERAGE,
        cmap="rainbow",
    )
    fig.colorbar(caxes, location="top", format="%.2f", pad=0.05)

    ax.set_xticks(range(len(dates)), dates, rotation=0)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel("Date", fontdict={"size": 22}, labelpad=13)
    ax.set_ylabel("Net Worth", fontdict={"size": 22}, labelpad=0)
    money = list(
        map(
            lambda x: f"${logToNumber(x) / 1e3:.1f}k".format(x),
            list(
                range(
                    MIN_VISUALIZE_STEPS,
                    MIN_VISUALIZE_STEPS + optimalPortfolioLeverageMatrix.shape[0],
                )
            ),
        )
    )
    ax.set_yticks(range(len(money)), money)
    ax.locator_params(axis="y", nbins=12)
    ax.locator_params(axis="x", nbins=6)
    ax.tick_params(axis="both", which="major", pad=15)
    plt.title("Optimal portfolio leverage over time")
    plt.grid(True)
    fig.tight_layout()
    os.makedirs("results/optimalPortfolioLeverage", exist_ok=True)
    plt.savefig(
        f"results/optimalPortfolioLeverage/{simulationResult.strategy.strategyName}.png"
    )


def visualizeAssetClassExposures(simulationResult: SimulationResult):
    (
        optimalEquityExposureMatrix,
        optimalBondExposureMatrix,
        optimalGoldExposureMatrix,
    ) = calculateAssetClassExposures(simulationResult)
    dates = list(simulationResult.balancePerDay.keys())

    optimalEquityExposureMatrix = optimalEquityExposureMatrix[
        :, MIN_VISUALIZE_STEPS:800
    ].transpose()
    optimalBondExposureMatrix = optimalBondExposureMatrix[
        :, MIN_VISUALIZE_STEPS:800
    ].transpose()
    optimalGoldExposureMatrix = optimalGoldExposureMatrix[
        :, MIN_VISUALIZE_STEPS:800
    ].transpose()

    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(18, 12))

    ax = plt.axes(projection="3d")

    (x, y) = np.meshgrid(
        np.arange(optimalEquityExposureMatrix.shape[1]),
        np.arange(optimalEquityExposureMatrix.shape[0]),
    )
    ax.plot_surface(
        x,
        y,
        optimalEquityExposureMatrix,
        color="r",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    ax.set_xlabel("Net Worth", fontdict={"size": 22}, labelpad=13)
    ax.set_ylabel("Date", fontdict={"size": 22}, labelpad=13)
    ax.set_zlabel("Exposure", fontdict={"size": 22}, labelpad=13)
    ax.set_title("Exposure over time")
    ax.plot_surface(
        x,
        y,
        optimalBondExposureMatrix,
        color="b",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    ax.plot_surface(
        x,
        y,
        optimalGoldExposureMatrix,
        color="y",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    plt.legend(["Equity", "Bonds", "Gold"])
    plt.grid(True)
    fig.tight_layout()
    os.makedirs("results/optimalAssetClassExposures", exist_ok=True)
    plt.savefig(
        f"results/optimalAssetClassExposures/{simulationResult.strategy.strategyName}.png"
    )


def visualizeAssetClassExposuresDifferentPlots(
    simulationResult: SimulationResult, normalize: bool = False
):
    (
        optimalEquityExposureMatrix,
        optimalBondExposureMatrix,
        optimalGoldExposureMatrix,
    ) = calculateAssetClassExposures(simulationResult)
    dates = list(simulationResult.balancePerDay.keys())

    optimalEquityExposureMatrix = optimalEquityExposureMatrix[
        :, MIN_VISUALIZE_STEPS:800
    ].transpose()
    optimalBondExposureMatrix = optimalBondExposureMatrix[
        :, MIN_VISUALIZE_STEPS:800
    ].transpose()
    optimalGoldExposureMatrix = optimalGoldExposureMatrix[
        :, MIN_VISUALIZE_STEPS:800
    ].transpose()

    plots = [
        (
            optimalEquityExposureMatrix,
            "Equity part of exposure" if normalize else "Equity leverage",
        ),
        (
            optimalBondExposureMatrix,
            "Bond part of exposure" if normalize else "Bond leverage",
        ),
        (
            optimalGoldExposureMatrix,
            "Gold part of exposure" if normalize else "Gold leverage",
        ),
    ]

    if normalize:
        totalExposure = (
            optimalEquityExposureMatrix
            + optimalBondExposureMatrix
            + optimalGoldExposureMatrix
        )
        optimalEquityExposureMatrix /= totalExposure
        optimalBondExposureMatrix /= totalExposure
        optimalGoldExposureMatrix /= totalExposure
        plots = [
            (totalExposure, "Total leverage"),
            *plots,
        ]

    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(ncols=1, nrows=len(plots), figsize=(18, 26))
    for i, (matrix, title) in enumerate(plots):
        caxes = ax[i].matshow(
            matrix,
            interpolation="bicubic",
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=3.0 if not normalize or i == 0 else np.max(matrix),
            cmap="rainbow" if not normalize or i == 0 else "viridis",
        )
        fig.colorbar(caxes, ax=ax[i], location="right", format="%.2f", pad=0.01)
        ax[i].set_xticks(range(len(dates)), dates, rotation=0)
        ax[i].xaxis.set_ticks_position("bottom")
        if i == len(plots) - 1:
            ax[i].set_xlabel("Date", fontdict={"size": 22}, labelpad=13)
        ax[i].set_ylabel("Net Worth", fontdict={"size": 22}, labelpad=0)
        money = list(
            map(
                lambda x: f"${logToNumber(x) / 1e3:.1f}k".format(x),
                list(
                    range(
                        MIN_VISUALIZE_STEPS,
                        MIN_VISUALIZE_STEPS + matrix.shape[0],
                    )
                ),
            )
        )
        ax[i].set_yticks(range(len(money)), money)
        ax[i].locator_params(axis="y", nbins=7)
        ax[i].locator_params(axis="x", nbins=6)
        ax[i].tick_params(axis="both", which="major", pad=15)
        ax[i].set_title(title, fontdict={"size": 22}, pad=13)

        plt.grid(True)
    fig.tight_layout()
    os.makedirs("results/optimalAssetClassExposures", exist_ok=True)
    plt.savefig(
        f"results/optimalAssetClassExposures/{simulationResult.strategy.strategyName}_{normalize}.png"
    )


def calculateAssetClassExposures(
    simulationResult: SimulationResult,
) -> np.ndarray:
    dp = simulationResult.strategy.dp
    numberOfDays = len(simulationResult.strategy.periodsToSim)
    optimalEquityExposureMatrix = np.zeros((numberOfDays, TOTAL_STEPS))
    optimalBondExposureMatrix = np.zeros((numberOfDays, TOTAL_STEPS))
    optimalGoldExposureMatrix = np.zeros((numberOfDays, TOTAL_STEPS))
    for day in tqdm(range(numberOfDays), desc="Calculating optimal portfolio leverage"):
        for money in range(TOTAL_STEPS):
            if dp[day][money] is None:
                continue
            _, bestPortfolio, _, _ = dp[day][money]
            optimalEquityExposureMatrix[day, money] = bestPortfolio.portfolio.get(
                "^GSPC", 0.0
            )
            optimalBondExposureMatrix[day, money] = bestPortfolio.portfolio.get(
                "TLT", 0.0
            )
            optimalGoldExposureMatrix[day, money] = bestPortfolio.portfolio.get(
                "GLD", 0.0
            )

    optimalEquityExposureMatrix = interpolateMatrix(
        optimalEquityExposureMatrix,
        newShapeY=simulationResult.strategy.daysPerRebalance
        * optimalEquityExposureMatrix.shape[0],
    )[: len(simulationResult.balancePerDay), :]
    optimalBondExposureMatrix = interpolateMatrix(
        optimalBondExposureMatrix,
        newShapeY=simulationResult.strategy.daysPerRebalance
        * optimalBondExposureMatrix.shape[0],
    )[: len(simulationResult.balancePerDay), :]
    optimalGoldExposureMatrix = interpolateMatrix(
        optimalGoldExposureMatrix,
        newShapeY=simulationResult.strategy.daysPerRebalance
        * optimalGoldExposureMatrix.shape[0],
    )[: len(simulationResult.balancePerDay), :]
    return (
        optimalEquityExposureMatrix,
        optimalBondExposureMatrix,
        optimalGoldExposureMatrix,
    )
