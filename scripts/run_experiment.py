import yaml
import time
import numpy as np
from pathlib import Path

from utils.math_utils import get_max_n
from src.transforms.matrix_to_spaces import get_YU_space
from src.transforms.matrix_to_meanings import get_true_meaning, generate_meaning_space, sample_meaning_space
from src.rewards.reward_func import reward_func
from src.agents.crsa_agent import CRSAAgent
from src.envs.negotiation_protocol import NegotiationProtocol
from src.envs.matrix_game import MatrixGame
from src.crsa.crsa import CRSA

start = time.time()

root = Path(__file__).resolve().parent.parent

def open_matrix_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env = config['env']
    matrix = config['matrix']

    game_name = env['name']
    game_type = env['type']
    num_actions = matrix['size']
    payoff_A = matrix['A']
    payoff_B = matrix['B']

    return game_name, game_type, num_actions, payoff_A, payoff_B

def open_crsa_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    crsa_params = config['crsa']

    return crsa_params

def run_experiment():
    # =====Open Config=====
    mat_config_path = root / 'configs' / 'matrices' / '15x15.yaml'
    game_name, game_type, num_actions, payoff_A, payoff_B = open_matrix_config(mat_config_path)
    payoff_A = np.array(payoff_A)
    payoff_B = np.array(payoff_B)

    params_config_path = root / 'configs' / 'crsa' / 'crsa_base.yaml'
    crsa_params = open_crsa_config(params_config_path)

    # =====Get CRSA Params=====
    reward_type = crsa_params['reward_type']
    y_opt = reward_func(reward_type, payoff_A, payoff_B)
    n = crsa_params.get('n', get_max_n(num_actions))
    n=15
    Y_space = get_YU_space(num_actions)
    U_space = set(Y_space)
    true_meaning_A = get_true_meaning(payoff_A, n)
    true_meaning_B = get_true_meaning(payoff_B, n)
    max_M = crsa_params.get('max_meaning_space_size', None)
    meaning_space = sample_meaning_space(
        num_actions, n, max_M,
        true_meanings=[true_meaning_A, true_meaning_B]
    ) if max_M else list(generate_meaning_space(num_actions, n))

    # =====Initiate Agents, Env, NegotiationProtocol=====
    agent_A = CRSAAgent("A", payoff_A, true_meaning_A, crsa_params['tau_A'])
    agent_B = CRSAAgent("B", payoff_B, true_meaning_B, crsa_params['tau_B'])
    game = MatrixGame(payoff_A, payoff_B, Y_space, y_opt, reward_type)
    crsa = CRSA(crsa_params['recursion_depth'], meaning_space)
    neg_protocol = NegotiationProtocol(game, agent_A, agent_B, crsa, U_space, crsa_params['turns'])

    print(agent_A.true_meaning)
    print(agent_B.true_meaning)
    print(game.y_opt)

    final_action, turns_taken, agreement = neg_protocol.run()

    pA = payoff_A.flatten()
    pB = payoff_B.flatten()

    print("\n" + "="*50)
    print("RÉSUMÉ DE LA NÉGOCIATION")
    print("="*50)
    print(f"  Accord atteint     : {agreement}")
    print(f"  Tours utilisés     : {turns_taken}")
    if final_action is not None:
        print(f"  Action choisie     : {final_action}  (row={final_action // num_actions}, col={final_action % num_actions})")
        print(f"  Payoff A           : {pA[final_action]:.3f}")
        print(f"  Payoff B           : {pB[final_action]:.3f}")
        print(f"  Payoff joint       : {pA[final_action] + pB[final_action]:.3f}")
    print(f"  Action optimale    : {y_opt}  (row={y_opt // num_actions}, col={y_opt % num_actions})")
    print(f"  Payoff A optimal   : {pA[y_opt]:.3f}")
    print(f"  Payoff B optimal   : {pB[y_opt]:.3f}")
    print(f"  Payoff joint opt.  : {pA[y_opt] + pB[y_opt]:.3f}")
    if final_action is not None:
        is_optimal = (final_action == y_opt)
        regret = (pA[y_opt] + pB[y_opt]) - (pA[final_action] + pB[final_action])
        print(f"  Action optimale ?  : {'OUI' if is_optimal else 'NON'}")
        print(f"  Regret (sans coût) : {regret:.3f}")
    if game.reward:
        print(f"  Reward net (coût inclus) : {game.reward[-1]:.3f}")
        print(f"  Regret total (coût inclus): {game.regret[-1]:.3f}")
    print("="*50)

    print(f"\nTemps d'exécution: {time.time() - start:.2f}s")

run_experiment()