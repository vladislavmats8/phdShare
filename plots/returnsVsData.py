import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gmean, gstd

from plots.plotUtils import (
    REAL_NAMES,
    STOCKS,
    TOLERANCE,
    USE_RETURNS,
    getCommonDates,
    getPrices,
)
from utils import LOCAL_PATH


def buildReturnsVsDataPlots(
    stockToCompare: str,
    dataToFilter: str,
    startDate,
    endDate,
):
    tradingDays = getCommonDates(startDate, endDate, [stockToCompare, dataToFilter])

    data = {
        stockToCompare: getPrices(stockToCompare, tradingDays),
        f"filter_{dataToFilter}": getPrices(dataToFilter, tradingDays),
    }

    df = pd.DataFrame(data, index=tradingDays)

    if dataToFilter in USE_RETURNS:
        df = df.pct_change()
    else:
        df[[stockToCompare]] = df[[stockToCompare]].pct_change()

    df = df.dropna()

    dataToFilterValues = sorted(set(sorted(df[f"filter_{dataToFilter}"])))
    x, ansMean, ansMedian, dest = [], [], [], []
    ansPlusStd, ansMinusStd = [], []

    for value in dataToFilterValues:
        df1 = df[
            ((value - TOLERANCE[dataToFilter]) <= df[f"filter_{dataToFilter}"])
            & (df[f"filter_{dataToFilter}"] <= value)
        ]

        if (len(df1) < 100) or (len(x) == 0 and len(df1) < 1000):
            continue

        valueToPush = df1[f"filter_{dataToFilter}"].mean()
        x.append(valueToPush * 100 if dataToFilter in USE_RETURNS else valueToPush)
        meanReturn = gmean(1 + df1[stockToCompare])
        stdReturn = (gstd(1 + df1[stockToCompare]) - 1) / np.sqrt(len(df1))
        ansMean.append(meanReturn**250 * 100 - 100)
        ansMedian.append((1 + df1[stockToCompare].median()) ** 250 * 100 - 100)
        ansPlusStd.append((meanReturn + stdReturn) ** 250 * 100 - 100)
        ansMinusStd.append((meanReturn - stdReturn) ** 250 * 100 - 100)
        dest.append(len(df1) / len(df))

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax2 = ax1.twinx()
    ax1.plot(x, ansMean, label="Average returns", color="red")
    ax1.plot(x, ansMedian, label="Median returns", color="green")
    ax1.plot(x, ansPlusStd, ":", linewidth=1, color="red")
    ax1.plot(x, ansMinusStd, ":", linewidth=1, color="red")
    ax2.plot(x, dest, "--", color="blue", label="Frequency")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)

    dataShowName = (
        f"{REAL_NAMES[dataToFilter]}"
        if dataToFilter not in USE_RETURNS
        else f"{REAL_NAMES[dataToFilter]} returns %,"
    )

    ax1.grid(True)
    ax1.set_xlabel(f"{dataShowName}")
    ax1.set_ylabel("Returns annualized %")
    ax2.set_ylabel("Frequency")
    plt.title(
        f"Average returns of {REAL_NAMES.get(stockToCompare, stockToCompare)}"
        + f"\nbased on {dataShowName} ({tradingDays[0]} - {tradingDays[-1]})"
    )
    plt.ylim(top=1)
    path = os.path.join(
        f"{LOCAL_PATH}",
        "results",
        "returns_vs_data",
        dataToFilter,
    )
    os.makedirs(path, exist_ok=True)
    plt.savefig(
        os.path.join(
            path,
            f"returns_vs_data_{stockToCompare}_{dataToFilter}",
        )
    )


def run():
    startDate = datetime.now().date() + timedelta(days=-300000)
    endDate = datetime.now().date() + timedelta(days=-3)

    for dataToFilter in [
        "fedFundsRate",
        "coreInflationUS",
        "vixIndex",
    ]:
        for stock in STOCKS + ["vixIndex"]:
            buildReturnsVsDataPlots(
                stockToCompare=stock,
                dataToFilter=dataToFilter,
                startDate=startDate,
                endDate=endDate,
            )


if __name__ == "__main__":
    run()
