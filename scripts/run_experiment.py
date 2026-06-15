import yaml
from pathlib import Path
from utils.math_utils import *
from src.transforms.matrix_to_spaces import *
from src.transforms.matrix_to_meanings import *

#create env
#create agents
#use protocol here

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
    mat_config_path = root / 'configs' / 'matrices' / '3x3.yaml'
    game_name, game_type, num_actions, payoff_A, payoff_B = open_matrix_config(mat_config_path)

    params_config_path = root / 'configs' / 'crsa' / 'crsa_base.yaml'
    crsa_params = open_crsa_config(params_config_path)

    n = get_max_n(num_actions)
    #TODO: need to decide on the n. The n obtained above is the max. Most probably should be smaller than that.
    Y_space = get_Y_space(num_actions)
    U_space = get_U_space(num_actions)
    true_meaning_A = get_true_meaning(payoff_A, n)
    tru_meaning_B = get_true_meaning(payoff_B, n)
    M_space = generate_meaning_space(num_actions, n)


run_experiment()