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


def sample_meaning_space_smart(k, n, max_size, y_opt, tau_A, tau_B, true_meanings=None, seed=42):
    """
    Échantillonnage biaisé vers les meanings compatibles avec y_opt.

    Chaque agent échantillonne les meanings du listener de son point de vue :
      - Agent A (speaker) → échantillonne des m_L avec tau_B comme seuil
      - Agent B (speaker) → échantillonne des m_L avec tau_A comme seuil

    Règle d'inclusion pour un meaning candidat m_cand :
      - cost = m_cand[y_opt]  (niveau de coût de l'action optimale selon ce meaning)
      - Si cost <= tau (compatible) → toujours inclus
      - Sinon → inclus avec probabilité = (n + 1 - cost) / n
                 (proportionnel à la récompense implicite de y_opt selon ce meaning)
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

    remaining = max_size - len(pool)
    budget_per_side = remaining // 2

    # A like speaker samples B's meanings (threshold = tau_B), and vice versa
    for tau in [tau_B, tau_A]:
        side_pool = set()
        attempts = 0
        max_attempts = max_size * 500

        while len(side_pool) < budget_per_side and attempts < max_attempts:
            m_cand = tuple(rng.integers(1, n + 1, size=dim).tolist())
            if m_cand in pool or m_cand in side_pool:
                attempts += 1
                continue

            cost = m_cand[y_opt]

            if cost <= tau:
                side_pool.add(m_cand)
            else:
                p = (n + 1 - cost) / n
                if rng.random() < p:
                    side_pool.add(m_cand)

            attempts += 1

        pool.update(side_pool)

    # Combler les places restantes aléatoirement
    attempts = 0
    while len(pool) < max_size and attempts < max_size * 10:
        pool.add(tuple(rng.integers(1, n + 1, size=dim).tolist()))
        attempts += 1

    return [np.array(m) for m in list(pool)[:max_size]]