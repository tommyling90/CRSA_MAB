import numpy as np
from src.priors.priors import y_prior_given_mS_mL, joint_prior, compatible
from src.priors.lexicon import lexicon_with_history

class CRSA:
    def __init__(self, recursion_depth, meaning_space):
        self.recursion_depth = recursion_depth
        self.meaning_space = meaning_space
        self.speaker_cache = {}
        self.belief_over_M = {}

    def choose_utterance(self, speaker, listener, game, U_space, turn, history):
        dist = self.get_speaker_dist(speaker.true_meaning, speaker.tau, listener.tau, U_space, game.Y_space, game.y_opt, turn, speaker.agent_id, history)
        u = np.random.choice(
            list(dist.keys()),
            p=list(dist.values())
        )
        return u

    def get_lit_listener_dist(self, m_L, tau_S, tau_L, u, Y_space, y_opt, w=None):
        if w is None:
            w = []

        values = {}
        Z = 0

        for y in Y_space:
            val = sum(
                joint_prior(m_S, m_L, tau_S, tau_L, y, y_opt) *
                lexicon_with_history(m_S, tau_S, u, w)
                for m_S in self.meaning_space
            )
            values[y] = val
            Z += val

        if Z == 0:
            return {y: 0 for y in Y_space}

        return {y: values[y] / Z for y in Y_space}

    def get_speaker_score(self, m_S, tau_S, tau_L, u, Y_space, y_opt, w, joint_belief_den, turn):
        # dummy value to avoid log(0)
        dummy = 1e-12
        score = 0.0

        for cand_m_L in self.meaning_space:
            y_dist = self.get_lit_listener_dist(cand_m_L, tau_S, tau_L, u, Y_space, y_opt, w)
            for y in Y_space:
                joint_prob = self.joint_belief(m_S, cand_m_L, tau_S, tau_L, y, y_opt, joint_belief_den)
                prob_y = y_dist[y]
                # TODO: need to think up a real cost func. Also how to make it dependent on utterance not just turns?
                # What would be the real value of c be?
                value = np.log(prob_y + dummy) - self.cost(turn)

                score += joint_prob * value

        return score

    def get_speaker_dist(self, m_S, tau_S, tau_L, U_space, Y_space, y_opt, turn, curr_agent, w, alpha=1.0):
        for cand_m in self.meaning_space:
            key = tuple(cand_m)
            self.belief_over_M[key] = self.belief(
                cand_m,
                turn,
                w,
                curr_agent
            )

        # calculer le dénominateur pour le belief conjoint ici pour éviter un loop non-nécessaire
        denominator = sum(
            self.belief_over_M[tuple(cand_m_L)]
            * compatible(m_S, cand_m_L, tau_S, tau_L, y_opt)
            for cand_m_L in self.meaning_space
        )

        if denominator == 0:
            raise RuntimeError("joint_belief denominator is zero")

        # commencer le calcul pour la distribution
        scores = {}

        for u in U_space:
            scores[u] = self.get_speaker_score(m_S, tau_S, tau_L, u, Y_space, y_opt, w, denominator, turn)

        # softmax: reduce size to prevent overflow (e.g. something like exp(1000))
        max_score = max(scores.values())
        unnorm = {
            u: np.exp(alpha * (scores[u] - max_score))
            for u in U_space
        }

        Z = sum(unnorm.values())
        if Z <= 0:
            raise RuntimeError("Normalized Speaker dist is zero")

        u_dist = {u: unnorm[u] / Z for u in U_space}

        key = (turn, tuple(m_S))
        self.speaker_cache[key] = u_dist

        print({"curr_agent": curr_agent, "meaning": tuple(m_S), "dist": u_dist})

        return u_dist

    def belief(self, cand_m_L, turn, w, curr_agent):
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
            key = (i, tuple(cand_m_L))
            #TODO: temp fix
            if key not in self.speaker_cache:
                prod *= 1
            else:
                prod *= self.speaker_cache[key][u_i]

        return prod

    def joint_belief(self, m_S, m_L, tau_S, tau_L, y, y_opt, joint_belief_den):
        key = tuple(m_L)
        return (
                self.belief_over_M[key]
                * compatible(m_S, m_L, tau_S, tau_L, y_opt)
                * y_prior_given_mS_mL(m_S, m_L, tau_S, tau_L, y, y_opt)
                / joint_belief_den
        )

    def cost(self, turn, c=0.05, free_turns=2):
        # free_turns = 2 since it requires at least a proposal and an acceptance to reach consensus
        # pick c = 0.05 arbitrarily small for now. Real cost func to be discussed.
        if turn <= free_turns:
            return 0.0
        # penalize when the negotiation drags on
        return c * (turn - free_turns)

