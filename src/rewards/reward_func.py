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

def penalty(turns, c=0.01, free_turns=2):
    # free_turns = 2 since it requires at least a proposal and an acceptance to reach consensus
    # pick c = 0.05 arbitrarily small for now. Real cost func to be discussed.
    if turns <= free_turns:
        return 0.0
    # penalize when the negotiation drags on
    return c * (turns - free_turns)