import math
from src.priors.priors import y_prior_given_mS_mL, compatible
from src.crsa.listener import get_lit_listener_dist
from src.transforms.matrix_to_meanings import generate_meaning_space

def get_speaker_score(m_S, tau_S, tau_L, u, Y_space, y_opt, k, n, w, joint_belief_den, Ml_space):
    #dummy value to avoid log(0)
    dummy = 1e-12
    score = 0.0

    for cand_m_L in Ml_space:
        y_dist = get_lit_listener_dist(cand_m_L, tau_S, tau_L, u, Y_space, y_opt, k, n, w)
        for y in Y_space:
            joint_prob = joint_belief(m_S, cand_m_L, tau_S, tau_L, y, y_opt, joint_belief_den, w)
            prob_y = y_dist[y]
            #TODO: need to think up a cost function in the value below
            value = math.log(prob_y + dummy)

            score += joint_prob * value

    return score

def get_speaker_dist(m_S, tau_S, tau_L, U_space, Y_space, y_opt, k, n, alpha=1.0, w=None):
    # calculer le dénominateur pour le belief conjoint ici pour éviter un loop non-nécessaire
    Ml_space = list(generate_meaning_space(k, n))
    denominator = sum(
        belief(w)
        * compatible(m_S, cand_m_L, tau_S, tau_L, y_opt)
        for cand_m_L in Ml_space
    )

    if denominator == 0:
        raise RuntimeError("joint_belief denominator is zero")

    # commencer le calcul pour la distribution
    scores = {}

    for u in U_space:
        scores[u] = get_speaker_score(m_S, tau_S, tau_L, u, Y_space, y_opt, k, n, w, denominator, Ml_space)

    #softmax: reduce size to prevent overflow (e.g. something like exp(1000))
    max_score = max(scores.values())
    unnorm = {
        u: math.exp(alpha * (scores[u] - max_score))
        for u in U_space
    }

    Z = sum(unnorm.values())
    if Z <= 0:
        raise RuntimeError("Normalized Speaker dist is zero")

    return { u: unnorm[u] / Z for u in U_space }

def belief(w):
    if not w:
        return 1
    #TODO: need to code in the real product terms here
    return

def joint_belief(m_S, m_L, tau_S, tau_L, y, y_opt, joint_belief_den, w):
    numerator = (
        belief(w)
        * compatible(m_S, m_L, tau_S, tau_L, y_opt)
        * y_prior_given_mS_mL(m_S, m_L, tau_S, tau_L, y, y_opt)
    )

    return numerator / joint_belief_den