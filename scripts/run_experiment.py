import yaml
import time
from pathlib import Path

from utils.math_utils import get_max_n
from src.transforms.matrix_to_spaces import get_Y_space, get_U_space
from src.transforms.matrix_to_meanings import get_true_meaning, generate_meaning_space
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
    mat_config_path = root / 'configs' / 'matrices' / '3x3.yaml'
    game_name, game_type, num_actions, payoff_A, payoff_B = open_matrix_config(mat_config_path)

    params_config_path = root / 'configs' / 'crsa' / 'crsa_base.yaml'
    crsa_params = open_crsa_config(params_config_path)

    # =====Get CRSA Params=====
    y_opt = reward_func(crsa_params['reward_type'], payoff_A, payoff_B)
    n = get_max_n(num_actions)
    #TODO: need to decide on the n. The n obtained above is the max. Most probably should be smaller than that.
    Y_space = get_Y_space(num_actions)
    U_space = get_U_space(num_actions)
    true_meaning_A = get_true_meaning(payoff_A, n)
    true_meaning_B = get_true_meaning(payoff_B, n)
    meaning_space = list(generate_meaning_space(num_actions, n))

    # =====Initiate Agents, Env, NegotiationProtocol=====
    agent_A = CRSAAgent("A", payoff_A, true_meaning_A, crsa_params['tau_A'])
    agent_B = CRSAAgent("B", payoff_B, true_meaning_B, crsa_params['tau_B'])
    game = MatrixGame(payoff_A, payoff_B, Y_space, y_opt)
    crsa = CRSA(crsa_params['recursion_depth'], meaning_space)
    neg_protocol = NegotiationProtocol(game, agent_A, agent_B, crsa, U_space, crsa_params['turns'])

    neg_protocol.run()

    print(time.time() - start)

run_experiment()