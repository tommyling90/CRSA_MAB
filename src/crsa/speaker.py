from src.priors.priors import y_prior_given_mS_mL, compatible
from src.transforms.matrix_to_meanings import generate_meaning_space

def get_speaker_dist():
    return

def belief(w):
    if not w:
        return 1
    #TODO: need to code in the real product terms here
    return

def joint_belief(m_S, m_L, tau_S, tau_L, y, y_opt, k, n, w=None):
    Ml_space = list(generate_meaning_space(k, n))

    numerator = (
        belief(w)
        * compatible(m_S, m_L, tau_S, tau_L, y_opt)
        * y_prior_given_mS_mL(m_S, m_L, tau_S, tau_L, y, y_opt)
    )

    denominator = sum(
        belief(w)
        * compatible(m_S, cand_m_L, tau_S, tau_L, y_opt)
        for cand_m_L in Ml_space
    )

    return numerator / denominator