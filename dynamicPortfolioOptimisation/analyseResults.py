import csv
import math
from scipy.stats import norm
from utils import LOCAL_PATH


def read_csv_files(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, "r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data.append(row)
    return data


def analyze_data(data):
    combinations = {}
    for row in data:
        utility_function_name = float(row[0])
        income_name = row[1]
        final_utility_only = int(row[2])
        if final_utility_only:
            continue

        utility_value = float(row[3].split(" ")[-1])
        matching_utility_value = float(row[4].split(" ")[-1])

        key = (utility_function_name, income_name, final_utility_only)
        if key not in combinations:
            combinations[key] = {
                "utility_gt_matching": 0,
                "utility_lt_matching": 0,
                "total_experiments": 0,
            }

        if utility_value > matching_utility_value:
            combinations[key]["utility_gt_matching"] += 1
        elif utility_value < matching_utility_value:
            combinations[key]["utility_lt_matching"] += 1
        combinations[key]["total_experiments"] += 1

    combinations[1e9, 1e9, 1e9] = {
        "utility_gt_matching": sum(
            value["utility_gt_matching"] for value in combinations.values()
        ),
        "utility_lt_matching": sum(
            value["utility_lt_matching"] for value in combinations.values()
        ),
        "total_experiments": sum(
            value["total_experiments"] for value in combinations.values()
        ),
    }

    table_data = []
    for key, value in combinations.items():
        utility_function_name, income_name, final_utility_only = key
        utility_gt_matching = value["utility_gt_matching"]
        utility_lt_matching = value["utility_lt_matching"]
        total_experiments = value["total_experiments"]
        proportion = utility_gt_matching / total_experiments
        z_score = (proportion - 0.5) / math.sqrt(0.5 * 0.5 / total_experiments)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        table_data.append(
            [
                utility_function_name,
                income_name,
                final_utility_only,
                utility_gt_matching,
                utility_lt_matching,
                total_experiments,
                proportion,
                z_score,
                p_value,
            ]
        )

    table_data.sort(key=lambda x: (x[0], x[1], x[2]))
    return table_data


def print_table(table_data):
    header = [
        "Utility Function Name",
        "Income Name",
        "Final Utility Only",
        "Utility > Matching Utility (Count)",
        "Utility < Matching Utility (Count)",
        "Total Experiments",
        "Proportion (Utility > Matching Utility)",
        "Z-Score",
        "P-Value",
    ]
    print("| " + " | ".join(header) + " |")
    print("|-" + "-|-".join(["-" * len(h) for h in header]) + "-|")
    for row in table_data:
        formatted_row = [
            str(value)[:6] if isinstance(value, (int, float)) else value
            for value in row
        ]
        print("| " + " | ".join(formatted_row) + " |")


if __name__ == "__main__":
    file_paths = [f"{LOCAL_PATH}/experimentResults.csv"]
    data = read_csv_files(file_paths)
    table_data = analyze_data(data)
    print_table(table_data)
