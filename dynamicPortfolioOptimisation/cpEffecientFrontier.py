import cvxpy as cp
import numpy as np

from dynamicPortfolioOptimisation.constants import ALLOW_SHORT, MAX_LEVERAGE


def find_efficient_frontier(means, stds, correlation):
    """
    Find the efficient frontier for a given set of assets using quadratic programming.

    Parameters:
    - means: A dictionary of expected returns for each asset.
    - stds: A dictionary of standard deviations for each asset.
    - correlation: A nested dictionary of correlation coefficients between each pair of assets.

    Returns:
    - efficient_frontier: A list of tuples (risk, return) representing the efficient frontier.
    """

    assets = list(means.keys())
    returns = np.array([means[asset] for asset in assets])
    std_devs = np.array([stds[asset] for asset in assets])
    corr_matrix = np.array(
        [[correlation[i].get(j, 1 if i == j else 0) for j in assets] for i in assets]
    )
    covariance_matrix = np.outer(std_devs, std_devs) * corr_matrix
    weights = cp.Variable(len(assets))
    portfolio_return = cp.sum(cp.multiply(returns, weights))
    portfolio_variance = cp.quad_form(weights, covariance_matrix)
    efficient_frontier = []

    for gamma in np.linspace(1e-1, 10, 201):
        # Objective: minimize variance - gamma * return
        objective = cp.Minimize(portfolio_variance - gamma * portfolio_return)
        constraints = [
            cp.sum(weights) == 1,
            MAX_LEVERAGE >= weights,
            weights[:-1] >= (-MAX_LEVERAGE if ALLOW_SHORT else 0.0),
            weights[-1] >= -MAX_LEVERAGE + 1,
        ]
        cp.Problem(objective, constraints).solve()

        if efficient_frontier and portfolio_return.value <= efficient_frontier[-1][1]:
            break

        efficient_frontier.append(
            (weights.value, portfolio_return.value, cp.sqrt(portfolio_variance).value)
        )

    return efficient_frontier


if __name__ == "__main__":
    means = {"Asset1": 0.1, "Asset2": 0.2, "Asset3": 0.15}
    stds = {"Asset1": 0.05, "Asset2": 0.1, "Asset3": 0.08}
    correlation = {
        "Asset1": {"Asset2": 0.5, "Asset3": 0.3},
        "Asset2": {"Asset1": 0.5, "Asset3": 0.6},
        "Asset3": {"Asset1": 0.3, "Asset2": 0.6},
    }

    efficient_frontier = find_efficient_frontier(means, stds, correlation)
    print(efficient_frontier)
