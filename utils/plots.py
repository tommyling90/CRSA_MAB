import csv
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGURE_DIR = PROJECT_ROOT / "experiments" / "figures"
RAW_DIR = PROJECT_ROOT / "experiments" / "raw"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

def save_top_belief_plot(rows, turn, curr_agent, output_dir=FIGURE_DIR):
    labels = [row["meaning_id"] for row in rows]
    probs = [row["prob"] for row in rows]

    plt.figure(figsize=(8, 6))
    plt.barh(labels[::-1], probs[::-1])
    plt.xlabel("Belief probability")
    plt.ylabel("Meaning index")
    plt.title(f"Turn {turn}: Agent {curr_agent}'s top beliefs")
    plt.tight_layout()

    path = f"{output_dir}/turn_{turn}_agent_{curr_agent}_beliefs.png"
    plt.savefig(path)
    plt.close()

    return path

def save_belief_legend(rows, turn, curr_agent, output_dir=RAW_DIR):
    path = f"{output_dir}/turn_{turn}_agent_{curr_agent}_legend.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "meaning_id", "prob", "meaning"])

        for row in rows:
            writer.writerow([
                row["rank"],
                row["meaning_id"],
                row["prob"],
                row["meaning"]
            ])

    return path