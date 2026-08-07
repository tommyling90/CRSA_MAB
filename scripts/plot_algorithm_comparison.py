from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from utils.plots import (
    RAW_DIR,
    save_agreement_rate_plot,
    save_mean_turns_plot,
)


RESULT_FILES = [
    next(RAW_DIR.glob("crsa_*_results.csv")),
    next(RAW_DIR.glob("greedy_*_results.csv")),
    next(RAW_DIR.glob("greedy_ii_*_results.csv")),
]


def parse_optional_float(value):
    if value in ("", "None", None):
        return None

    return float(value)


def parse_optional_int(value):
    if value in ("", "None", None):
        return None

    return int(value)


def parse_bool(value):
    return str(value).lower() == "true"


def load_results():
    results = []

    for path in RESULT_FILES:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing result file: {path}"
            )

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                results.append({
                    "algorithm": row["algorithm"],
                    "episode": int(row["episode"]),
                    "agreement": parse_bool(
                        row["agreement"]
                    ),
                    "final_u": parse_optional_int(
                        row["final_u"]
                    ),
                    "turns": int(row["turns"]),
                    "is_optimal": parse_bool(
                        row["is_optimal"]
                    ),
                    "payoff_A": parse_optional_float(
                        row["payoff_A"]
                    ),
                    "payoff_B": parse_optional_float(
                        row["payoff_B"]
                    )
                })

    return results


def main():
    results = load_results()

    comparison_name = "crsa_vs_greedy_vs_greedy_ii_3x3_epi10"

    save_agreement_rate_plot(
        results,
        comparison_name,
    )

    save_mean_turns_plot(
        results,
        comparison_name,
    )

    print(
        "Saved comparison graphs to "
        "experiments/figures/"
    )


if __name__ == "__main__":
    main()