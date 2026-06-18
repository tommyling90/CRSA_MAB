import math
from src.priors.priors import y_prior_given_mS_mL, compatible
from src.crsa.listener import get_lit_listener_dist
from src.transforms.matrix_to_meanings import generate_meaning_space

def get_speaker_score(m_S, tau_S, tau_L, u, Y_space, y_opt, k, n, w, joint_belief_den, Ml_space, turn, belief_over_M):
    #dummy value to avoid log(0)
    dummy = 1e-12
    score = 0.0

    for cand_m_L in Ml_space:
        y_dist = get_lit_listener_dist(cand_m_L, tau_S, tau_L, u, Y_space, y_opt, k, n, w)
        for y in Y_space:
            joint_prob = joint_belief(m_S, cand_m_L, tau_S, tau_L, y, y_opt, joint_belief_den, belief_over_M)
            prob_y = y_dist[y]
            #TODO: need to think up a real cost func. Also how to make it dependent on utterance not just turns?
            # What would be the real value of c be?
            value = math.log(prob_y + dummy) - cost(turn)

            score += joint_prob * value

    return score

def get_speaker_dist(m_S, tau_S, tau_L, U_space, Y_space, y_opt, k, n, turn, speaker_cache, curr_agent, alpha=1.0, w=None):
    Ml_space = list(generate_meaning_space(k, n))

    belief_over_M = {
        cand_m: belief(cand_m, turn, speaker_cache, w, curr_agent)
        for cand_m in Ml_space
    }

    # calculer le dénominateur pour le belief conjoint ici pour éviter un loop non-nécessaire
    denominator = sum(
        belief_over_M[cand_m_L]
        * compatible(m_S, cand_m_L, tau_S, tau_L, y_opt)
        for cand_m_L in Ml_space
    )

    if denominator == 0:
        raise RuntimeError("joint_belief denominator is zero")

    # commencer le calcul pour la distribution
    scores = {}

    for u in U_space:
        scores[u] = get_speaker_score(m_S, tau_S, tau_L, u, Y_space, y_opt, k, n, w, denominator, Ml_space, turn, belief_over_M)

    #softmax: reduce size to prevent overflow (e.g. something like exp(1000))
    max_score = max(scores.values())
    unnorm = {
        u: math.exp(alpha * (scores[u] - max_score))
        for u in U_space
    }

    Z = sum(unnorm.values())
    if Z <= 0:
        raise RuntimeError("Normalized Speaker dist is zero")

    u_dist = { u: unnorm[u] / Z for u in U_space }

    key = (turn, m_S)
    speaker_cache[key] = u_dist

    return u_dist

def belief(cand_m_L, turn, speaker_cache, w, curr_agent):
    if not w:
        return 1.0
    prod = 1.0

    other_agent = "B" if curr_agent == "A" else "A"

    for i, event in enumerate(w):
        if i >= turn:
            break
        if event["speaker"] != other_agent:
            continue

        u_i = event["utterance"]
        key = (i, cand_m_L)
        prod *= speaker_cache[key][u_i]

    return prod

def joint_belief(m_S, m_L, tau_S, tau_L, y, y_opt, joint_belief_den, belief_over_M):
    return (
        belief_over_M[m_L]
        * compatible(m_S, m_L, tau_S, tau_L, y_opt)
        * y_prior_given_mS_mL(m_S, m_L, tau_S, tau_L, y, y_opt)
        / joint_belief_den
    )

def cost(turn, c=0.05, free_turns=2):
    # free_turns = 2 since it requires at least a proposal and an acceptance to reach consensus
    # pick c = 0.05 arbitrarily small for now. Real cost func to be discussed.
    if turn <= free_turns:
        return 0.0
    # penalize when the negotiation drags on
    return c*(turn - free_turns)