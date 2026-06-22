import numpy as np

def reward_func(type, payoff_A, payoff_B):
    y_opt = -1

    if type == 'utilitarian':
        y_opt = np.argmax(payoff_A + payoff_B)

    return y_opt

def calc_reward(type, payoff_A, payoff_B, action):
    if type == 'utilitarian':
        return payoff_A[action] + payoff_B[action]
    return