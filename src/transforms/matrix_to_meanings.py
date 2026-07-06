import numpy as np

def get_true_meaning(payoff, n, tau):
    #evenly divide the range by n then collapse the ranks where rank > tau into tau+1
    max_val = payoff.max()

    normalized = payoff / max_val
    levels = np.ceil(normalized * n).astype(int)
    ranks = n + 1 - levels

    #collapse here
    true_meaning = np.where(ranks > tau, tau + 1, ranks)

    return true_meaning.flatten()

def generate_meaning_space(k, n, dtype=np.uint8):
    #permutations
    size = k * k
    total = n ** size

    M = np.empty((total, size), dtype=dtype)

    for col in range(size):
        repeat = n ** (size - col - 1)
        tile = n ** col
        M[:, col] = np.tile(
            np.repeat(np.arange(1, n + 1, dtype=dtype), repeat),
            tile
        )

    return M