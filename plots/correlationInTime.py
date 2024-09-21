import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd

from plots.plotUtils import REAL_NAMES, STOCKS, getCommonDates, getPrices
from utils import LOCAL_PATH

COLORMAP = {
    "b": {"marker": None, "dash": (None, None)},
    "g": {"marker": None, "dash": [5, 5]},
    "r": {"marker": None, "dash": [5, 3, 1, 3]},
    "c": {"marker": None, "dash": [1, 3]},
    "m": {"marker": None, "dash": [5, 2, 5, 2, 5, 10]},
    "y": {"marker": None, "dash": [5, 3, 1, 2, 1, 10]},
    "k": {"marker": "o", "dash": (None, None)},
}


def buildCorrelationPlotsInTime(
    stockToCompare: str,
    stockCompareAgainst: str,
    startDate,
    endDate,
):
    tradingDays = getCommonDates(
        startDate, endDate, [stockToCompare, stockCompareAgainst]
    )

    data = {
        stockToCompare: getPrices(stockToCompare, tradingDays),
        stockCompareAgainst: getPrices(stockCompareAgainst, tradingDays),
    }
    df = pd.DataFrame(data, index=tradingDays)
    df_returns = df.pct_change().dropna()

    corr10 = (
        df_returns[stockToCompare].rolling(10).corr(df_returns[stockCompareAgainst])
    ).dropna()
    corr50 = (
        df_returns[stockToCompare].rolling(50).corr(df_returns[stockCompareAgainst])
    ).dropna()
    corr250 = (
        df_returns[stockToCompare].rolling(250).corr(df_returns[stockCompareAgainst])
    ).dropna()

    correlation = df_returns.corr()

    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(figsize=(9, 6))

    plt.plot(corr10, label="10 days rolling correlation", color="#00006c", marker=None)
    plt.grid(True)
    plt.plot(corr50, label="50 days rolling correlation", color="#00ec00", marker=None)
    plt.plot(corr250, label="250 days rolling correlation", color="#ffdb00")
    ymin = min(list(corr10))
    xpos = list(corr10).index(ymin)

    plt.annotate(
        "10 days",
        (corr10.index[xpos], ymin),
        xycoords="data",
        xytext=(-100, -10),
        annotation_clip=False,
        textcoords="offset pixels",
        arrowprops=dict(arrowstyle="-|>"),
    )
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.legend()
    name = stockCompareAgainst
    plt.title(
        f"Correlation between {REAL_NAMES.get(stockToCompare, stockToCompare)} and {REAL_NAMES.get(name, name)}, average = {correlation.iloc[0][1]:.2f}"
    )
    fig.tight_layout()
    plt.savefig(
        os.path.join(
            f"{LOCAL_PATH}/results/correlationsInTime",
            f"correlation_in_time_{name}_{stockToCompare}",
        )
    )

    print(f"Correlation coefficient:\n{correlation}")


def run():
    stockToCompare = "^GSPC"
    startDate = datetime.now().date() + timedelta(days=-300000)
    endDate = datetime.now().date() + timedelta(days=-3)
    for stockCompareAgainst in STOCKS + ["vixIndex"]:
        buildCorrelationPlotsInTime(
            stockToCompare=stockToCompare,
            stockCompareAgainst=stockCompareAgainst,
            startDate=startDate,
            endDate=endDate,
        )
