import numpy as np

class CRSA:
    def __init__(self, recursion_depth, meaning_spaces):
        self.recursion_depth = recursion_depth
        self.meaning_spaces = meaning_spaces
        self.speaker_cache = {}
        self._l0_cache = {}  # cache: (w_utterances) → (|M|, |U|)

    def choose_utterance(self, speaker, listener, game, U_space, turn, history):
        # self.precompute_speaker_cache(
        #     speaker.tau,
        #     listener.tau,
        #     U_space,
        #     game.Y_space,
        #     game.y_opt,
        #     turn,
        #     history
        # )

        M_L = np.array(self.meaning_spaces[listener.agent_id])
        M_S = np.array(self.meaning_spaces[speaker.agent_id])
        dist = self.get_speaker_dist(speaker.true_meaning, speaker.tau, listener.tau, U_space,  game.y_opt, turn, speaker.agent_id, history, M_L, M_S)
        u = np.random.choice(
            list(dist.keys()),
            p=list(dist.values())
        )
        return u

    #Notez que dans cette matrice c'est si y_opt a la prob de 1 ou 0 etant donné un certain m_L et u.
    #les autres y ne sont pas pertinents pcq c'est forcément 0 (selon les formules de prior et lexicon)
    def _l0_matrix(self, M_S, M_L, tau_S, tau_L, U_arr, y_opt, w):
        utterances = [event["utterance"] for event in w]
        last_u = utterances[-1] if utterances else None

        #TODO: pk on a besoin de cache???
        cache_key = tuple(utterances)
        if cache_key in self._l0_cache:
            return self._l0_cache[cache_key]

        # vectorisation below. Basically decompose the formula of lit listener and multiply the composites
        lex_hist = np.array([
            0 if (u in utterances and u != last_u) else 1
            for u in U_arr
        ], dtype=float)  # (|U|,)

        compat_mS = (M_S[:, y_opt] <= tau_S)  # (|M_S|,) bool
        compat_mL = (M_L[:, y_opt] <= tau_L)  # (|M_L|,) bool
        lex_S = (M_S[:, U_arr] <= tau_S)  # (|M_S|, |U|) bool

        sum_lex = (compat_mS[:, None] * lex_S).sum(axis=0)  # (|U|,)

        L0_unnorm = compat_mL[:, None].astype(float) * (sum_lex * lex_hist)[None, :]  # (|M_L|, |U|)
        L0 = np.where(L0_unnorm > 0, 1.0, 0.0)
        self._l0_cache[cache_key] = L0
        return L0

    def get_speaker_dist(self, m_S, tau_S, tau_L, U_space, y_opt, turn, curr_agent, w, M_L, M_S, alpha=1.0):
        U_arr = np.array(sorted(U_space))

        # Calcul des beliefs sur tout le meaning space M_L
        beliefs = np.array([self.belief(M_L[j], turn, w, curr_agent) for j in range(len(M_L))])

        # Normaliser pour éviter l'underflow numérique sur de nombreux tours
        b_max = beliefs.max()
        if b_max > 0:
            beliefs = beliefs / b_max
        else:
            beliefs = np.ones(len(M_L))

        # Dénominateur du belief conjoint: sum_mL B(mL) * compat(mS, mL)
        compat_mL = (M_L[:, y_opt] <= tau_L).astype(float)  # (|M|,)
        m_S_compat = float(m_S[y_opt] <= tau_S)
        denominator = float((beliefs * compat_mL).sum()) * m_S_compat

        if denominator == 0:
            raise RuntimeError("joint_belief denominator is zero")

        # L0 vectorisé et mis en cache pour ce tour
        L0 = self._l0_matrix(M_S, M_L, tau_S, tau_L, U_arr, y_opt, w)  # (|M|, |U|)

        joint_beliefs = beliefs * compat_mL * m_S_compat / denominator  # (|M_L|,)

        dummy = 1e-12
        # TODO: need to come up with a cost function (should be a part of prior)
        # here we got the score - the sum across y and M_L for each u
        scores_vec = joint_beliefs @ np.log(L0 + dummy)  # (|U|,)

        # Filtre lexique: le speaker ne propose que des actions dans son propre lexique
        valid_mask = L0.any(axis=0)
        scores_vec = np.where(valid_mask, scores_vec, -np.inf)

        if not valid_mask.any():
            raise RuntimeError("No valid utterances in speaker lexicon for this meaning")

        # Softmax stable (soustrait le max pour éviter overflow)
        max_score = scores_vec[valid_mask].max()
        unnorm = np.where(valid_mask, np.exp(alpha * (scores_vec - max_score)), 0.0)
        Z = unnorm.sum()
        if Z <= 0:
            raise RuntimeError("Normalized Speaker dist is zero")

        probs = unnorm / Z
        u_dist = {int(u): float(probs[k]) for k, u in enumerate(U_arr)}

        self.speaker_cache[(turn, tuple(m_S))] = u_dist
        print({"curr_agent": curr_agent, "meaning": tuple(m_S), "dist": u_dist})
        return u_dist

    #TODO: thought i could precompute the speaker_cache but turned out too costly. Ask Lautaro what they did?

    # def precompute_speaker_cache(self, tau_S, tau_L, U_space, Y_space, y_opt, turn, w):
    #     for i, event in enumerate(w):
    #         if i >= turn:
    #             break
    #         past_speaker = event["speaker"]
    #
    #         for cand_m in self.meaning_space:
    #             key = (i, tuple(cand_m))
    #
    #             if key not in self.speaker_cache:
    #                 self.speaker_cache[key] = self.get_speaker_dist(
    #                     cand_m,
    #                     tau_S,
    #                     tau_L,
    #                     U_space,
    #                     Y_space,
    #                     y_opt,
    #                     i,
    #                     past_speaker,
    #                     w[:i]
    #                 )

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
            key = (other_agent, i, tuple(cand_m_L))
            #TODO: need to find a way to calculate for speaker_cache where key not exists
            if key not in self.speaker_cache:
                prod *= 1
            else:
                prod *= self.speaker_cache[key][u_i]

        return prod

   # relic functions below - may not be used at all
   #  def get_lit_listener_dist(self, m_L, tau_S, tau_L, u, Y_space, y_opt, M_S, w=None):
   #      if w is None:
   #          w = []
   #
   #      values = {}
   #      Z = 0
   #
   #      for y in Y_space:
   #          val = sum(
   #              joint_prior(m_S, m_L, tau_S, tau_L, y, y_opt) *
   #              lexicon_with_history(m_S, tau_S, u, w)
   #              for m_S in M_S
   #          )
   #          values[y] = val
   #          Z += val
   #
   #      if Z == 0:
   #          return {y: 0 for y in Y_space}
   #
   #      return {y: values[y] / Z for y in Y_space}
   #
   #  def get_speaker_score(self, m_S, tau_S, tau_L, u, Y_space, y_opt, w, joint_belief_den, M_L, M_S):
   #      # dummy value to avoid log(0)
   #      dummy = 1e-12
   #      score = 0.0
   #
   #      for cand_m_L in M_L:
   #          y_dist = self.get_lit_listener_dist(cand_m_L, tau_S, tau_L, u, Y_space, y_opt, M_S, w)
   #          for y in Y_space:
   #              joint_prob = self.joint_belief(m_S, cand_m_L, tau_S, tau_L, y, y_opt, joint_belief_den)
   #              prob_y = y_dist[y]
   #              value = np.log(prob_y + dummy)
   #
   #              score += joint_prob * value
   #
   #      return score

    # def joint_belief(self, m_S, m_L, tau_S, tau_L, y, y_opt, joint_belief_den):
    #     key = tuple(m_L)
    #     return (
    #             self.belief_over_M[key]
    #             * compatible(m_S, m_L, tau_S, tau_L, y_opt)
    #             * y_prior_given_mS_mL(m_S, m_L, tau_S, tau_L, y, y_opt)
    #             / joint_belief_den
    #     )