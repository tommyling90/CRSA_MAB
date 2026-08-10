import numpy as np

def reward_func(type, meaning_A, meaning_B):
    meaning_A = np.asarray(meaning_A)
    meaning_B = np.asarray(meaning_B)

    if type == "utilitarian":
        scores = meaning_A + meaning_B
        best_score = scores.min()

        y_opts = np.flatnonzero(scores == best_score)

        return [int(y) for y in y_opts]

    raise ValueError(f"Unknown reward type: {type}")

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