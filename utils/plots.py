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
def _meaning_label(meaning):
    """Compact display of a flattened meaning, e.g. 123122313."""
    return "".join(str(int(x)) for x in meaning)


def save_selected_episode_belief_heatmap(
    belief_history,
    speaker_history,
    meaning_spaces,
    agent_id,
    run_name,
    episode_number,
    top_k=20,
    output_dir=FIGURE_DIR,
):
    """
    Belief heatmap for ONE episode only.

    For Agent A, rows are candidate meanings of B.
    For Agent B, rows are candidate meanings of A.
    """

    rows = sorted(
        [
            row for row in belief_history
            if row["agent"] == agent_id
        ],
        key=lambda row: row["turn"],
    )

    if not rows:
        return None

    other_agent = "B" if agent_id == "A" else "A"
    M_other = np.asarray(meaning_spaces[other_agent])

    turns = [row["turn"] for row in rows]

    all_probs = np.stack([
        np.asarray(row["probs"])
        for row in rows
    ])

    # Use one fixed set of meanings for the whole heatmap:
    # meanings that reached the highest probability at any turn.
    max_probs = all_probs.max(axis=0)

    top_indices = np.argsort(
        max_probs
    )[-top_k:][::-1]

    matrix = all_probs[:, top_indices].T

    # Current utterance at each turn
    utterance_by_turn = {
        row["turn"]: row["chosen_u"]
        for row in speaker_history
    }

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
    )

    # x-axis: turn + actual utterance
    ax.set_xticks(
        np.arange(len(turns))
    )

    ax.set_xticklabels([
        f"Turn {turn}\nu={utterance_by_turn.get(turn, '?')}"
        for turn in turns
    ])

    # y-axis: ACTUAL meaning, not m_12345
    ax.set_yticks(
        np.arange(len(top_indices))
    )

    ax.set_yticklabels([
        _meaning_label(M_other[idx])
        for idx in top_indices
    ])

    ax.set_xlabel("Turn / utterance")
    ax.set_ylabel(
        f"Candidate meaning of Agent {other_agent}"
    )

    ax.set_title(
        f"CRSA belief — Agent {agent_id}, episode{episode_number}"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Belief probability",
    )

    fig.tight_layout()

    path = (
        Path(output_dir)
        / f"{run_name}_selected_episode{episode_number}_belief_heatmap_agent_{agent_id}.png"
    )

    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_selected_episode_speaker_heatmap(
    speaker_history,
    agent_id,
    episode_number,
    run_name,
    output_dir=FIGURE_DIR,
):
    """
    Speaker distribution for ONE episode only.
    """

    rows = sorted(
        [
            row for row in speaker_history
            if row["agent"] == agent_id
        ],
        key=lambda row: row["turn"],
    )

    if not rows:
        return None

    utterances = sorted(
        rows[0]["dist"].keys()
    )

    turns = [
        row["turn"]
        for row in rows
    ]

    matrix = np.array([
        [
            row["dist"][u]
            for u in utterances
        ]
        for row in rows
    ])

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
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

    # turn + actual sampled utterance
    ax.set_yticklabels([
        f"Turn {row['turn']}  →  u={row['chosen_u']}"
        for row in rows
    ])

    ax.set_xlabel(
        "Possible utterance / joint action"
    )
    ax.set_ylabel(
        "Turn / actual utterance"
    )

    ax.set_title(
        f"CRSA speaker distribution — Agent {agent_id}, episode{episode_number}"
    )

    # Mark actual sampled utterance with X
    u_to_col = {
        u: i
        for i, u in enumerate(utterances)
    }

    for row_idx, row in enumerate(rows):
        ax.scatter(
            u_to_col[row["chosen_u"]],
            row_idx,
            marker="x",
            s=90,
        )

    fig.colorbar(
        image,
        ax=ax,
        label="Proposal probability",
    )

    fig.tight_layout()

    path = (
        Path(output_dir)
        / f"{run_name}_selected_episode{episode_number}_speaker_heatmap_agent_{agent_id}.png"
    )

    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_selected_episode_top20_beliefs(
    belief_history,
    speaker_history,
    meaning_spaces,
    agent_id,
    run_name,
    episode_number,
    top_k=20,
    output_dir=FIGURE_DIR,
):
    """
    One simple horizontal-bar figure PER TURN,
    containing that turn's top-20 beliefs.
    """

    rows = sorted(
        [
            row for row in belief_history
            if row["agent"] == agent_id
        ],
        key=lambda row: row["turn"],
    )

    if not rows:
        return

    other_agent = (
        "B" if agent_id == "A" else "A"
    )

    M_other = np.asarray(
        meaning_spaces[other_agent]
    )

    utterance_by_turn = {
        row["turn"]: row["chosen_u"]
        for row in speaker_history
    }

    for row in rows:

        turn = row["turn"]
        probs = np.asarray(
            row["probs"]
        )

        top_indices = np.argsort(
            probs
        )[-top_k:][::-1]

        top_probs = probs[top_indices]

        labels = [
            _meaning_label(
                M_other[idx]
            )
            for idx in top_indices
        ]

        # Reverse so highest appears at top
        labels = labels[::-1]
        top_probs = top_probs[::-1]

        fig, ax = plt.subplots(
            figsize=(9, 7)
        )

        ax.barh(
            labels,
            top_probs,
        )

        utterance = utterance_by_turn.get(
            turn,
            "?"
        )

        ax.set_xlabel(
            "Belief probability"
        )

        ax.set_ylabel(
            f"Candidate meaning of Agent {other_agent}"
        )

        ax.set_title(
            f"Agent {agent_id} top {top_k} beliefs — "
            f"Turn {turn}, utterance u={utterance}"
        )

        fig.tight_layout()

        path = (
            Path(output_dir)
            / (
                f"{run_name}_selected_episode{episode_number}_"
                f"turn{turn}_agent_{agent_id}_top{top_k}_beliefs.png"
            )
        )

        fig.savefig(
            path,
            dpi=300,
        )

        plt.close(fig)


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

def save_listener_accuracy_plot(
    listener_y_histories,
    run_name,
    output_dir=FIGURE_DIR,
):
    """
    Listener top-1 accuracy at each REAL turn.

    At turn t, only episodes that genuinely reached turn t
    contribute to the mean.
    """

    if not listener_y_histories:
        return None

    by_turn = {}

    for episode_history in listener_y_histories:
        for row in episode_history:

            turn = row["turn"]

            by_turn.setdefault(
                turn,
                []
            ).append(
                float(row["correct"])
            )

    turns = sorted(by_turn)

    accuracies = [
        np.mean(
            by_turn[turn]
        )
        for turn in turns
    ]

    counts = [
        len(
            by_turn[turn]
        )
        for turn in turns
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        turns,
        accuracies,
        marker="o",
    )

    # Show how many genuine episodes contribute at each turn
    for turn, accuracy, count in zip(
        turns,
        accuracies,
        counts,
    ):
        ax.annotate(
            f"n={count}",
            (turn, accuracy),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    ax.set_xlabel("Turn")

    ax.set_ylabel(
        "Listener top-1 accuracy"
    )

    ax.set_title(
        "CRSA listener accuracy among active negotiations"
    )

    ax.set_ylim(0, 1.05)
    ax.set_xticks(turns)

    ax.grid(alpha=0.3)

    fig.tight_layout()

    path = (
        Path(output_dir)
        / f"{run_name}_listener_accuracy.png"
    )

    fig.savefig(
        path,
        dpi=300,
    )

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
                "payoff_B",
                "y_opts",
                "payoff_matrix_A",
                "payoff_matrix_B",
                "true_meaning_A",
                "true_meaning_B",
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