import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGURE_DIR = PROJECT_ROOT / "experiments" / "figures"
RAW_DIR = PROJECT_ROOT / "experiments" / "raw"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CRSA
# ============================================================

def save_meaning_map(
    meaning_spaces,
    true_meanings,
    run_name,
    matrix_size,
    output_dir=RAW_DIR,
):
    path = Path(output_dir) / f"{run_name}_meaning_map.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "agent",
            "meaning_id",
            "is_true_meaning",
            "meaning_flat",
            "meaning_matrix",
        ])

        for agent_id in ("A", "B"):
            meanings = np.asarray(meaning_spaces[agent_id])
            true_meaning = np.asarray(true_meanings[agent_id])

            for idx, meaning in enumerate(meanings):
                matrix = meaning.reshape(
                    matrix_size,
                    matrix_size,
                )

                writer.writerow([
                    agent_id,
                    f"m_{idx}",
                    bool(
                        np.array_equal(
                            meaning,
                            true_meaning
                        )
                    ),
                    meaning.tolist(),
                    matrix.tolist(),
                ])

    return path


def _mean_beliefs_by_turn(
    episode_histories,
    agent_id,
):
    """
    Returns mean posterior for each turn, averaging over
    episodes that actually reached that turn.
    """
    by_turn = {}

    for history in episode_histories:
        for row in history:
            if row["agent"] != agent_id:
                continue

            by_turn.setdefault(
                row["turn"],
                []
            ).append(
                np.asarray(row["probs"])
            )

    mean_by_turn = {}

    for turn, arrays in by_turn.items():
        mean_by_turn[turn] = np.mean(
            np.stack(arrays),
            axis=0,
        )

    return mean_by_turn


def save_mean_belief_heatmap(
    episode_histories,
    agent_id,
    run_name,
    top_k=10,
    output_dir=FIGURE_DIR,
):
    mean_by_turn = _mean_beliefs_by_turn(
        episode_histories,
        agent_id,
    )

    if not mean_by_turn:
        return None

    turns = sorted(mean_by_turn)

    all_probs = np.stack([
        mean_by_turn[turn]
        for turn in turns
    ])

    # Fixed meaning identities across all turns.
    max_prob = all_probs.max(axis=0)

    top_indices = np.argsort(
        max_prob
    )[-top_k:][::-1]

    matrix = all_probs[:, top_indices].T

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
    )

    ax.set_xticks(
        np.arange(len(turns))
    )

    ax.set_xticklabels(turns)

    ax.set_yticks(
        np.arange(len(top_indices))
    )

    ax.set_yticklabels([
        f"m_{idx}"
        for idx in top_indices
    ])

    ax.set_xlabel("Turn")
    ax.set_ylabel("Opponent meaning")
    ax.set_title(
        f"CRSA mean belief over meanings — Agent {agent_id}"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Mean belief probability",
    )

    fig.tight_layout()

    path = (
        Path(output_dir)
        / f"{run_name}_mean_belief_heatmap_agent_{agent_id}.png"
    )

    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_mean_entropy_plot(
    episode_histories,
    run_name,
    output_dir=FIGURE_DIR,
):
    by_agent_turn = {
        "A": {},
        "B": {},
    }

    for history in episode_histories:
        for row in history:
            agent = row["agent"]
            turn = row["turn"]

            probs = np.asarray(
                row["probs"],
                dtype=float,
            )

            positive = probs[
                probs > 0
            ]

            entropy = -np.sum(
                positive
                * np.log(positive)
            )

            max_entropy = np.log(
                len(probs)
            )

            normalized = (
                entropy / max_entropy
                if max_entropy > 0
                else 0
            )

            by_agent_turn[
                agent
            ].setdefault(
                turn,
                []
            ).append(normalized)

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for agent_id in ("A", "B"):
        turns = sorted(
            by_agent_turn[agent_id]
        )

        if not turns:
            continue

        means = [
            np.mean(
                by_agent_turn[
                    agent_id
                ][turn]
            )
            for turn in turns
        ]

        ax.plot(
            turns,
            means,
            marker="o",
            label=f"Agent {agent_id}",
        )

    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    ax.set_xlabel("Turn")
    ax.set_ylabel(
        "Mean normalized belief entropy"
    )

    ax.set_ylim(0, 1.05)

    ax.set_title(
        "CRSA mean belief uncertainty over episodes"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        Path(output_dir)
        / f"{run_name}_mean_belief_entropy.png"
    )

    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_mean_speaker_heatmap(
    episode_histories,
    agent_id,
    run_name,
    output_dir=FIGURE_DIR,
):
    by_turn = {}

    for history in episode_histories:
        for row in history:
            if row["agent"] != agent_id:
                continue

            by_turn.setdefault(
                row["turn"],
                []
            ).append(row["dist"])

    if not by_turn:
        return None

    turns = sorted(by_turn)

    first_turn = turns[0]
    first_dist = by_turn[first_turn][0]

    utterances = sorted(
        first_dist.keys()
    )

    matrix = []

    for turn in turns:
        turn_dists = by_turn[turn]

        mean_dist = [
            np.mean([
                dist[u]
                for dist in turn_dists
            ])
            for u in utterances
        ]

        matrix.append(mean_dist)

    matrix = np.asarray(matrix)

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
    )

    ax.set_xticks(
        np.arange(len(utterances))
    )

    ax.set_xticklabels(
        utterances
    )

    ax.set_yticks(
        np.arange(len(turns))
    )

    ax.set_yticklabels(
        turns
    )

    ax.set_xlabel(
        "Utterance / joint action"
    )

    ax.set_ylabel("Turn")

    ax.set_title(
        f"CRSA mean speaker distribution — Agent {agent_id}"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Mean proposal probability",
    )

    fig.tight_layout()

    path = (
        Path(output_dir)
        / f"{run_name}_mean_speaker_heatmap_agent_{agent_id}.png"
    )

    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


# ============================================================
# EXPERIMENT RESULTS
# ============================================================

def save_experiment_results(
    results,
    run_name,
    output_dir=RAW_DIR,
):
    if not results:
        return None

    path = (
        Path(output_dir)
        / f"{run_name}_results.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "episode",
                "agreement",
                "final_u",
                "turns",
                "is_optimal",
                "payoff_A",
                "payoff_B"
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    return path


# ============================================================
# ALGORITHM COMPARISON
# ============================================================

def save_agreement_rate_plot(
    results,
    comparison_name,
    output_dir=FIGURE_DIR,
):
    algorithms = sorted({
        row["algorithm"]
        for row in results
    })

    agreement_rates = []
    optimal_rates = []

    for algorithm in algorithms:
        rows = [
            row for row in results
            if row["algorithm"] == algorithm
        ]

        agreement_rates.append(
            np.mean([
                row["agreement"]
                for row in rows
            ]) * 100
        )

        optimal_rates.append(
            np.mean([
                row["is_optimal"]
                for row in rows
            ]) * 100
        )

    x = np.arange(
        len(algorithms)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x - width / 2,
        agreement_rates,
        width,
        label="Agreement",
    )

    ax.bar(
        x + width / 2,
        optimal_rates,
        width,
        label="Optimal agreement",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)

    ax.set_ylabel("Mean rate (%)")

    ax.set_title(
        "Mean agreement performance"
    )

    ax.set_ylim(0, 105)
    ax.legend()

    fig.tight_layout()

    path = (
        Path(output_dir)
        / f"{comparison_name}_agreement_rates.png"
    )

    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path

def save_mean_turns_plot(
    results,
    comparison_name,
    output_dir=FIGURE_DIR,
):
    algorithms = sorted({
        row["algorithm"]
        for row in results
    })

    means = []

    for algorithm in algorithms:
        values = [
            row["turns"]
            for row in results
            if (
                row["algorithm"] == algorithm
                and row["agreement"]
            )
        ]

        means.append(
            np.mean(values)
            if values
            else np.nan
        )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        algorithms,
        means,
    )

    ax.set_xlabel("Algorithm")

    ax.set_ylabel(
        "Mean turns to agreement"
    )

    ax.set_title(
        "Mean negotiation length"
    )

    fig.tight_layout()

    path = (
        Path(output_dir)
        / f"{comparison_name}_mean_turns_to_agreement.png"
    )

    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path