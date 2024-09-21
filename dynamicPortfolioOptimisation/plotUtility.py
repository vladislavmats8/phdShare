import matplotlib.pyplot as plt
import numpy as np


# Define the utility function
def utility_function(W_T, W_ref, gamma):
    return -((1 - np.minimum(np.log(W_T) / np.log(W_ref), 1)) ** gamma) + 1


def utility_function_scaled_both(W_T, W_ref, gamma, W_T_min, W_T_max):
    # Find the minimum utility value for W_T_min
    U_min = -((1 - min(np.log(W_T_min) / np.log(W_ref), 1)) ** gamma) + 1
    # Find the maximum utility value for W_T_max
    U_max = -((1 - min(np.log(W_T_max) / np.log(W_ref), 1)) ** gamma) + 1
    # Calculate the scaled utility value to start all plots at the same point and end at the same point
    return (utility_function(W_T, W_ref, gamma) - U_min) / (U_max - U_min)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 22})

    W_ref = 1e8  # Reference wealth
    W_T_min = 1e4  # Minimum terminal wealth
    W_T_max = 1e8  # Maximum terminal wealth
    gammas = range(1, 11)

    W_T_values = np.geomspace(W_T_min, W_T_max, 100)
    plt.figure(figsize=(18, 12))

    for gamma in gammas:
        U_values = utility_function_scaled_both(
            W_T_values, W_ref, gamma, W_T_min, W_T_max
        )
        plt.plot(W_T_values, U_values, label=f"γ = {gamma}")

    plt.title("Utility Function U(W) for Different γ Values", fontsize=22)
    plt.xlabel("Terminal Wealth W", fontsize=22)
    plt.ylabel("Utility U(W)", fontsize=22)
    plt.xscale("log", base=10)
    plt.legend()
    plt.grid(True)
    plt.show()
