from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

from dynamicPortfolioOptimisation.constants import (
    LOG_BASE,
    MONEY_BASE,
    NEGATIVE_STEPS,
    PROBABILITY_THR,
    TOTAL_STEPS,
)


def numberToLog(number: float) -> float:
    return np.log(number / MONEY_BASE) / np.log(LOG_BASE) + NEGATIVE_STEPS


def logToNumber(log: float) -> float:
    return LOG_BASE ** (log - NEGATIVE_STEPS) * MONEY_BASE


numberToLogArray = np.vectorize(numberToLog)
logToNumberArray = np.vectorize(logToNumber)


@dataclass
class Distribution:
    moneyBase: float = MONEY_BASE
    logBase: float = LOG_BASE
    TOTAL_STEPS: int = TOTAL_STEPS
    NEGATIVE_STEPS: int = NEGATIVE_STEPS
    probabilities: np.ndarray | None = None

    portfolio: Dict[str, float] = None

    def __post_init__(self):
        self.probabilities = (
            self.probabilities
            if self.probabilities is not None
            else np.zeros(self.TOTAL_STEPS)
        )

    def mean(self) -> float:
        return np.dot(np.arange(self.TOTAL_STEPS), self.probabilities)

    def meanAbsolute(self) -> float:
        return np.dot(logToNumberArray(np.arange(self.TOTAL_STEPS)), self.probabilities)

    def std(self) -> float:
        indices = np.arange(self.TOTAL_STEPS)
        return np.sqrt(np.dot((indices - self.mean()) ** 2, self.probabilities))

    def stdAbsolute(self) -> float:
        values = logToNumberArray(np.arange(self.TOTAL_STEPS))
        return np.sqrt(np.dot((values - self.meanAbsolute()) ** 2, self.probabilities))

    def getSignificantProbabilities(self) -> List[Tuple[int, float]]:
        return zip(
            np.where(self.probabilities > PROBABILITY_THR)[0],
            self.probabilities[self.probabilities > PROBABILITY_THR]
            / sum(self.probabilities[self.probabilities > PROBABILITY_THR]),
        )

    def print(self) -> None:
        for idx, probability in enumerate(self.probabilities):
            if probability > PROBABILITY_THR:
                money = self.moneyBase * (self.logBase ** (idx - self.NEGATIVE_STEPS))
                print(f"{money}: {probability}")
                print(idx)


def utility(stdMult: float, distribution: Distribution) -> float:
    probabilities = distribution.probabilities
    indices = np.arange(distribution.TOTAL_STEPS)
    meanValue = np.dot(indices, probabilities)
    standardDeviation = np.sqrt(np.dot((indices - meanValue) ** 2, probabilities))

    return meanValue - stdMult * standardDeviation


def utilityAbsolute(stdMult: float, distribution: Distribution) -> float:
    probabilities = distribution.probabilities
    indices = np.arange(distribution.TOTAL_STEPS)
    values = logToNumberArray(indices)
    meanValue = np.dot(values, probabilities)
    standardDeviation = np.sqrt(np.dot((values - meanValue) ** 2, probabilities))

    return meanValue - stdMult * standardDeviation


def getLognormDistribution(mean: float, std: float) -> Distribution:
    x = logToNumberArray(np.arange(TOTAL_STEPS) + 0.5) / MONEY_BASE
    cdf_values = stats.lognorm.cdf(x, s=std, scale=np.exp(mean))
    pdf_values = np.diff(cdf_values, prepend=0.0)
    pdf_values /= sum(pdf_values)
    dist = Distribution(probabilities=pdf_values)

    return dist
