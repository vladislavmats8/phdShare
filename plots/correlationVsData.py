import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd

from plots.plotUtils import (
    REAL_NAMES,
    STOCKS,
    TOLERANCE,
    USE_RETURNS,
    getCommonDates,
    getPrices,
    pearsonr_ci,
)
from utils import LOCAL_PATH


def buildCorrelationVsDataPlots(
    stockToCompare: str,
    stockCompareAgainst: str,
    dataToFilter: str,
    startDate,
    endDate,
    cumulative=False,
):
    tradingDays = getCommonDates(
        startDate, endDate, [stockToCompare, stockCompareAgainst, dataToFilter]
    )

    data = {
        stockToCompare: getPrices(stockToCompare, tradingDays),
        stockCompareAgainst: getPrices(stockCompareAgainst, tradingDays),
        f"filter_{dataToFilter}": getPrices(dataToFilter, tradingDays),
    }

    df = pd.DataFrame(data, index=tradingDays)
    if dataToFilter in USE_RETURNS:
        df = df.pct_change()
    else:
        df[[stockToCompare, stockCompareAgainst]] = df[
            [stockToCompare, stockCompareAgainst]
        ].pct_change()

    df = df.dropna()

    dataToFilterValues = sorted(set(sorted(df[f"filter_{dataToFilter}"])))
    x, correlations, dest = [], [], []
    correlationsLow, correlationsHigh = [], []

    for value in dataToFilterValues:
        df1 = df[
            ((value - TOLERANCE[dataToFilter]) <= df[f"filter_{dataToFilter}"])
            & (df[f"filter_{dataToFilter}"] <= value)
        ]

        if (len(df1) < 100) or (len(x) == 0 and len(df1) < 1000):
            continue

        correlation = df1[[stockToCompare, stockCompareAgainst]].corr().iloc[0, 1]
        correlation, low, high = pearsonr_ci(
            x=df1[stockCompareAgainst], y=df1[stockToCompare]
        )
        valueToPush = df1[f"filter_{dataToFilter}"].mean()
        x.append(valueToPush * 100 if dataToFilter in USE_RETURNS else valueToPush)
        correlations.append(correlation)
        correlationsLow.append(low)
        correlationsHigh.append(high)
        dest.append(len(df1) / len(df))

    fig, ax1 = plt.subplots(figsize=(9, 8))
    plt.rcParams.update({"font.size": 18})

    ax2 = ax1.twinx()
    ax1.plot(x, correlations, label="Correlation", color="#ff0000")
    ax1.plot(
        x,
        correlationsLow,
        ":",
        linewidth=3,
        label="Fisher p=0.99 confidence interval",
        color="#00ff00",
    )
    ax1.plot(x, correlationsHigh, ":", linewidth=3, color="#00ff00")
    ax2.plot(x, dest, "--", color="#0000ff", label="Frequency")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
    )

    ax1.grid(True)
    ax1.set_ylabel("Correlation")
    ax2.set_ylabel("Frequency")
    ax2.set_ylim(0, 1)
    ax1.set_fontsize(20)
    ax2.set_fontsize(20)

    dataShowName = (
        f"{REAL_NAMES[dataToFilter]}"
        if dataToFilter not in USE_RETURNS
        else f"{REAL_NAMES[dataToFilter]} returns %,"
    )

    plt.xticks = x
    ax1.grid(True)
    ax1.set_xlabel(f"{dataShowName}")
    plt.title(
        f"Correlation between {REAL_NAMES.get(stockCompareAgainst, stockCompareAgainst)} and {REAL_NAMES.get(stockToCompare, stockToCompare)}"
        + f"\nbased on {dataShowName} ({tradingDays[0]} - {tradingDays[-1]})"
    )
    path = f"{LOCAL_PATH}/correlation_vs_data/{dataToFilter}/cumulative={cumulative}"
    fig.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(
        os.path.join(
            path,
            f"correlation_vs_value_{stockToCompare}_{stockCompareAgainst}_{dataToFilter}",
        )
    )


def run():
    stockToCompare = "^GSPC"
    startDate = datetime.now().date() + timedelta(days=-300000)
    endDate = datetime.now().date() + timedelta(days=-3)
    for dataToFilter in [
        "^GSPC",
        "fedFundsRate",
        "cpiTotal",
        "coreInflationUS",
        "vixIndex",
    ]:
        for stockCompareAgainst in STOCKS + ["vixIndex"]:
            if stockCompareAgainst != stockToCompare:
                buildCorrelationVsDataPlots(
                    stockToCompare=stockToCompare,
                    stockCompareAgainst=stockCompareAgainst,
                    dataToFilter=dataToFilter,
                    startDate=startDate,
                    endDate=endDate,
                )


run()
