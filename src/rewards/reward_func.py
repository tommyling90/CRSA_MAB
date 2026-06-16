import numpy as np

def reward_func(type, payoff_A, payoff_B):
    payoff_A = np.array(payoff_A)
    payoff_B = np.array(payoff_B)
    y_opt = -1

    if type == 'utilitarian':
        y_opt = np.argmax(payoff_A + payoff_B)

    return y_opt