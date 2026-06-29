import numpy as np
from itertools import product

def get_true_meaning(payoff, n, tau):
    #evenly divide the range by n then collapse the ranks where rank > tau into tau+1
    max_val = payoff.max()

    normalized = payoff / max_val
    levels = np.ceil(normalized * n).astype(int)
    ranks = n + 1 - levels

    #collapse here
    true_meaning = np.where(ranks > tau, tau + 1, ranks)

    return true_meaning.flatten()

def generate_meaning_space(k, n):
    #permutations
    for values in product(range(1, n+1), repeat=k * k):
        #use yield to not waste memory
        yield np.array(values)