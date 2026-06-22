import numpy as np
from itertools import product

def get_true_meaning(payoff, n):
    #produce true meaning m* with naive method: evenly divide the range by n
    max_val = payoff.max()

    normalized = payoff / max_val
    levels = np.ceil(normalized * n).astype(int)
    true_meaning = n + 1 - levels

    return true_meaning.flatten()

def generate_meaning_space(k, n):
    #permutations
    for values in product(range(1, n+1), repeat=k * k):
        #use yield to not waste memory
        yield np.array(values)