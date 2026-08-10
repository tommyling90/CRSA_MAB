"""Command-line entry point for negotiation experiments.

Examples:
    python scripts/run_experiment.py --algorithm crsa --matrix 3x3
    python scripts/run_experiment.py -a greedy --episodes 10
    python scripts/run_experiment.py -a greedy_ii --turns 8 --tau-a 2 --tau-b 3
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from utils.plots import (
    save_experiment_results,
    save_meaning_map,
    save_mean_belief_heatmap,
    save_mean_entropy_plot,
    save_mean_speaker_heatmap,
)

import numpy as np
import yaml

# Running ``python scripts/run_experiment.py`` puts scripts/, rather than the
# repository root, on sys.path.  Add the root before importing project modules.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.crsa_agent import CRSAAgent
from src.algos.crsa import CRSA
from src.algos.greedy import Greedy
from src.algos.greedy_ii import GreedyII
from src.envs.matrix_game import MatrixGame
from src.envs.negotiation_protocol import NegotiationProtocol
from src.rewards.reward_func import reward_func
from src.transforms.matrix_to_meanings import get_true_meaning, generate_meaning_space
from src.transforms.matrix_to_spaces import get_YU_space
from utils.math_utils import get_max_n


ALGORITHMS = {
    "crsa": CRSA,
    "greedy": Greedy,
    "greedy_ii": GreedyII,
}


def _load_yaml(config_path: Path) -> dict[str, Any]:
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as exc:
        raise ValueError(f"Could not open config file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping")
    return config


def open_matrix_config(config_path: str | Path):
    config_path = Path(config_path)
    config = _load_yaml(config_path)
    try:
        env = config["env"]
        matrix = config["matrix"]
        game_name = env["name"]
        game_type = env["type"]
        num_actions = int(matrix["size"])
        payoff_A = np.asarray(matrix["A"], dtype=float)
        payoff_B = np.asarray(matrix["B"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed matrix config {config_path}: {exc}") from exc

    expected_shape = (num_actions, num_actions)
    if payoff_A.shape != expected_shape or payoff_B.shape != expected_shape:
        raise ValueError(
            f"Matrices A and B in {config_path} must both have shape "
            f"{expected_shape}; got {payoff_A.shape} and {payoff_B.shape}"
        )
    if num_actions < 1:
        raise ValueError("Matrix size must be at least 1")
    return game_name, game_type, num_actions, payoff_A, payoff_B


def open_crsa_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_yaml(config_path)
    if not isinstance(config.get("crsa"), dict):
        raise ValueError(f"Config file {config_path} must contain a 'crsa' mapping")
    return dict(config["crsa"])


def _config_path(value: str, directory: Path) -> Path:
    """Accept either a path or a config stem such as ``3x3``."""
    path = Path(value).expanduser()
    if path.exists() or path.suffix in {".yaml", ".yml"} or path.parent != Path("."):
        return path.resolve()
    return directory / f"{value}.yaml"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a CRSA_MAB matrix-game negotiation experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-a", "--algorithm", "--algo", choices=ALGORITHMS, default="crsa",
        help="negotiation algorithm",
    )
    parser.add_argument(
        "-m", "--matrix", "--matrix-config", dest="matrix_config", default="3x3",
        help="matrix config path, or a name from configs/matrices",
    )
    parser.add_argument(
        "-c", "--config", "--params-config", dest="params_config", default="crsa_base",
        help="parameter config path, or a name from configs/crsa",
    )
    parser.add_argument("--recursion-depth", type=_positive_int, help="CRSA recursion depth")
    parser.add_argument("--turns", type=_positive_int, help="maximum negotiation turns")
    parser.add_argument("--episodes", type=_positive_int, help="number of episodes")
    parser.add_argument("--n-a", dest="n_A", type=_positive_int, help="number of meaning ranks for A")
    parser.add_argument("--n-b", dest="n_B", type=_positive_int, help="number of meaning ranks for B")
    parser.add_argument("--tau-a", dest="tau_A", type=_positive_int, help="acceptability threshold for A")
    parser.add_argument("--tau-b", dest="tau_B", type=_positive_int, help="acceptability threshold for B")
    parser.add_argument("--alpha", type=float, help="CRSA softmax rationality")
    parser.add_argument("--reward-type", choices=("utilitarian",), help="joint reward function")
    parser.add_argument("--seed", type=int, help="NumPy random seed for reproducible runs")
    return parser


def _apply_overrides(params: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "recursion_depth", "turns", "episodes", "n_A", "n_B",
        "tau_A", "tau_B", "alpha", "reward_type",
    )
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            params[name] = value
    return params


def _validate_params(params: dict[str, Any], algorithm: str) -> dict[str, Any]:
    required = ["turns", "episodes", "tau_A", "tau_B", "reward_type"]
    if algorithm == "crsa":
        required.extend(("recursion_depth", "alpha"))
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError(f"Missing experiment parameter(s): {', '.join(missing)}")

    integer_params = ["turns", "episodes", "tau_A", "tau_B"]
    if algorithm == "crsa":
        integer_params.append("recursion_depth")
    for name in integer_params:
        params[name] = int(params[name])
        if params[name] < 1:
            raise ValueError(f"{name} must be at least 1")
    for name in ("n_A", "n_B"):
        if params.get(name) is not None:
            params[name] = int(params[name])
            if params[name] < 1:
                raise ValueError(f"{name} must be at least 1")
    if algorithm == "crsa":
        params["alpha"] = float(params["alpha"])
        if params["alpha"] <= 0:
            raise ValueError("alpha must be greater than 0")
    if params["reward_type"] != "utilitarian":
        raise ValueError(f"Unsupported reward type: {params['reward_type']}")
    return params


def _make_algorithm(
    name: str,
    params: dict[str, Any],
    meaning_spaces: dict[str, np.ndarray] | None,
    taus: dict[str, int],
):
    if name == "crsa":
        assert meaning_spaces is not None
        return CRSA(params["recursion_depth"], meaning_spaces, taus, params["alpha"])
    return ALGORITHMS[name]()


def run_experiment(args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Run an experiment from parsed CLI arguments and return a small summary."""
    if args is None:
        args = build_parser().parse_args()

    crsa_only = {
        "--recursion-depth": getattr(args, "recursion_depth", None),
        "--alpha": getattr(args, "alpha", None),
        "--seed": getattr(args, "seed", None),
    }
    incompatible = [option for option, value in crsa_only.items() if value is not None]
    if args.algorithm != "crsa" and incompatible:
        raise ValueError(
            f"{', '.join(incompatible)} can only be used with --algorithm crsa"
        )

    started_at = time.time()
    matrix_path = _config_path(args.matrix_config, ROOT / "configs" / "matrices")
    params_path = _config_path(args.params_config, ROOT / "configs" / "crsa")
    game_name, game_type, num_actions, payoff_A, payoff_B = open_matrix_config(matrix_path)
    params = _validate_params(
        _apply_overrides(open_crsa_config(params_path), args), args.algorithm
    )

    if args.seed is not None:
        np.random.seed(args.seed)

    n_A = params.get("n_A") or get_max_n(num_actions)
    n_B = params.get("n_B") or get_max_n(num_actions)
    tau_A, tau_B = params["tau_A"], params["tau_B"]
    if tau_A > n_A or tau_B > n_B:
        raise ValueError(
            f"tau values cannot exceed their meaning ranks "
            f"(tau_A={tau_A}, n_A={n_A}; tau_B={tau_B}, n_B={n_B})"
        )

    y_space = get_YU_space(num_actions)
    u_space = set(y_space)
    # First transform the payoff matrices.
    true_meaning_A = get_true_meaning(
        payoff_A,
        n_A,
        tau_A
    )

    true_meaning_B = get_true_meaning(
        payoff_B,
        n_B,
        tau_B
    )

    # Then determine ALL optimal y* from the transformed rankings.
    y_opts = reward_func(
        params["reward_type"],
        true_meaning_A,
        true_meaning_B,
    )
    taus = {"A": tau_A, "B": tau_B}

    matrix_name = matrix_path.stem

    seed_name = (
        args.seed
        if args.seed is not None
        else "na"
    )

    run_name = (
        f"{args.algorithm}_"
        f"{matrix_name}_"
        f"epi{params['episodes']}_"
        f"seed{seed_name}"
    )

    # Meaning-space enumeration is expensive and only CRSA consumes it.
    meaning_spaces = None
    if args.algorithm == "crsa":
        meaning_spaces = {
            "A": generate_meaning_space(num_actions, tau_A + 1 if tau_A < n_A else tau_A),
            "B": generate_meaning_space(num_actions, tau_B + 1 if tau_B < n_B else tau_B),
        }
        save_meaning_map(
            meaning_spaces=meaning_spaces,
            true_meanings={
                "A": true_meaning_A,
                "B": true_meaning_B,
            },
            run_name=run_name,
            matrix_size=num_actions,
        )

    agent_A = CRSAAgent("A", payoff_A, true_meaning_A, tau_A)
    agent_B = CRSAAgent("B", payoff_B, true_meaning_B, tau_B)
    game = MatrixGame(
        payoff_A, payoff_B, y_space, y_opts, params["reward_type"], params["episodes"]
    )

    print(
        f"Running {args.algorithm} on {game_name} ({game_type}), "
        f"matrix={matrix_path.name}, episodes={params['episodes']}, seed={args.seed}"
    )
    print(f"n_A={n_A}, n_B={n_B}, tau_A={tau_A}, tau_B={tau_B}, y_opt={y_opts}")
    print(true_meaning_A, true_meaning_B)

    agreements = 0
    results = []
    belief_histories = []
    speaker_histories = []

    for episode in range(params["episodes"]):
        print(f"\n=== EPISODE {episode + 1} ===")
        algorithm = _make_algorithm(args.algorithm, params, meaning_spaces, taus)
        protocol = NegotiationProtocol(
            game, agent_A, agent_B, algorithm, u_space, params["turns"]
        )
        final_u, turns, agreement = protocol.run()
        agreements += int(agreement)
        if args.algorithm == "crsa":
            belief_histories.append(
                algorithm.belief_history
            )

            speaker_histories.append(
                algorithm.speaker_history
            )

        if final_u is not None:
            payoff_A_final = float(
                payoff_A.flatten()[final_u]
            )

            payoff_B_final = float(
                payoff_B.flatten()[final_u]
            )

        else:
            payoff_A_final = None
            payoff_B_final = None

        results.append({
            "algorithm": args.algorithm,
            "episode": episode + 1,
            "agreement": agreement,
            "final_u": final_u,
            "turns": turns,
            "is_optimal": (
                    agreement
                    and final_u in y_opts
            ),
            "payoff_A": payoff_A_final,
            "payoff_B": payoff_B_final
        })
        is_last_episode = episode == params["episodes"] - 1
        agent_A.end_episode(final_u, print_stats=is_last_episode)
        agent_B.end_episode(final_u, print_stats=is_last_episode)

    elapsed = time.time() - started_at
    print(f"\nCompleted in {elapsed:.3f}s; agreements={agreements}/{params['episodes']}")
    save_experiment_results(
        results=results,
        run_name=run_name,
    )
    if args.algorithm == "crsa":

        for agent_id in ("A", "B"):
            save_mean_belief_heatmap(
                episode_histories=belief_histories,
                agent_id=agent_id,
                run_name=run_name,
            )

            save_mean_speaker_heatmap(
                episode_histories=speaker_histories,
                agent_id=agent_id,
                run_name=run_name,
            )

        save_mean_entropy_plot(
            episode_histories=belief_histories,
            run_name=run_name,
        )
    return {
        "algorithm": args.algorithm,
        "episodes": params["episodes"],
        "agreements": agreements,
        "results": results,
        "elapsed_seconds": elapsed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_experiment(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
