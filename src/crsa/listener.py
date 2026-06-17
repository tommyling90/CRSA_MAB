from src.priors.priors import joint_prior
from src.priors.lexicon import lexicon_with_history
from src.transforms.matrix_to_meanings import generate_meaning_space

def get_lit_listener_dist(m_L, tau_S, tau_L, u, Y_space, y_opt, k, n, w=None):
    if w is None:
        w = []

    values = {}
    Z = 0

    for y in Y_space:
        Ms_space_gen = generate_meaning_space(k, n)
        val = sum(
            joint_prior(m_S, m_L, tau_S, tau_L, y, y_opt) *
            lexicon_with_history(m_S, tau_S, u, w)
            for m_S in Ms_space_gen
        )
        values[y] = val
        Z += val

    if Z == 0:
        return {y: 0 for y in Y_space}

    return {y: values[y] / Z for y in Y_space}
