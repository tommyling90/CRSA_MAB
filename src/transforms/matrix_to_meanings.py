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
    for values in product(range(1, n+1), repeat=k * k):
        yield np.array(values)

def sample_meaning_space(k, n, max_size, true_meanings=None, seed=42):
    """
    Retourne au plus max_size meanings. Si l'espace complet <= max_size, l'énumère en entier.
    Sinon, échantillonne aléatoirement en garantissant que true_meanings sont inclus.
    """
    dim = k * k
    full_size = n ** dim

    if full_size <= max_size:
        return list(generate_meaning_space(k, n))

    rng = np.random.default_rng(seed)
    pool = set()

    if true_meanings is not None:
        for m in true_meanings:
            pool.add(tuple(int(v) for v in m))

    while len(pool) < max_size:
        sample = tuple(rng.integers(1, n + 1, size=dim).tolist())
        pool.add(sample)

    return [np.array(m) for m in pool]