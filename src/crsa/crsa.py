import numpy as np
from src.priors.priors import y_prior_given_mS_mL, joint_prior, compatible
from src.priors.lexicon import lexicon_with_history

class CRSA:
    def __init__(self, recursion_depth, meaning_space):
        self.recursion_depth = recursion_depth
        self.meaning_space = meaning_space
        self.M = np.array(meaning_space)  # (|M|, k²) — precomputed once
        self.speaker_cache = {}
        self.belief_over_M = {}
        self._l0_cache = {}  # cache: (tau_S, tau_L, y_opt, U_tuple, w_utterances) → (|M|, |U|)

    def choose_utterance(self, speaker, listener, game, U_space, turn, history):
        dist = self.get_speaker_dist(
            speaker.true_meaning, speaker.tau, listener.tau,
            U_space, game.Y_space, game.y_opt, turn, speaker.agent_id, history
        )
        u = np.random.choice(list(dist.keys()), p=list(dist.values()))
        return u

    def _l0_matrix(self, tau_S, tau_L, U_arr, y_opt, w):
        """
        Calcule L0(y_opt | m_L=M[j], u=U_arr[k]) pour tous (j, k) en une passe vectorisée.
        Résultat mis en cache. Shape: (|M|, |U|).
        """
        utterances = [e["utterance"] for e in w]
        last_u = utterances[-1] if utterances else None

        cache_key = (float(tau_S), float(tau_L), int(y_opt), tuple(U_arr.tolist()), tuple(utterances))
        if cache_key in self._l0_cache:
            return self._l0_cache[cache_key]

        # Contrainte historique: lex_hist[k] = 0 si U_arr[k] a déjà été dit (sauf le dernier)
        lex_hist = np.array([
            0 if (u in utterances and u != last_u) else 1
            for u in U_arr
        ], dtype=float)  # (|U|,)

        compat_mS = (self.M[:, y_opt] <= tau_S)      # (|M|,) bool
        compat_mL = (self.M[:, y_opt] <= tau_L)      # (|M|,) bool
        lex_S     = (self.M[:, U_arr] <= tau_S)      # (|M|, |U|) bool

        # Pour chaque u: somme sur m_S de compat_mS[i] * lex_S[i, u]
        sum_lex = (compat_mS[:, None] * lex_S).sum(axis=0)  # (|U|,)

        # L0_unnorm[j, k] = compat_mL[j] * sum_lex[k] * lex_hist[k]
        L0_unnorm = compat_mL[:, None].astype(float) * (sum_lex * lex_hist)[None, :]  # (|M|, |U|)

        # Seul y_opt a de la masse → après normalisation L0 vaut 1 ou 0
        L0 = np.where(L0_unnorm > 0, 1.0, 0.0)

        self._l0_cache[cache_key] = L0
        return L0

    def get_lit_listener_dist(self, m_L, tau_S, tau_L, u, Y_space, y_opt, w=None):
        if w is None:
            w = []
        U_arr = np.array(sorted(Y_space))
        L0 = self._l0_matrix(tau_S, tau_L, U_arr, y_opt, w)
        j_arr = np.where((self.M == m_L).all(axis=1))[0]
        if len(j_arr) == 0:
            return {y: 0 for y in Y_space}
        k = int(np.where(U_arr == u)[0][0])
        val = float(L0[j_arr[0], k])
        return {y: (val if y == y_opt else 0.0) for y in Y_space}

    def get_speaker_dist(self, m_S, tau_S, tau_L, U_space, Y_space, y_opt, turn, curr_agent, w, alpha=1.0):
        U_arr = np.array(sorted(U_space))

        # Calcul des beliefs sur tout le meaning space
        beliefs = np.array([self.belief(self.M[j], turn, w, curr_agent) for j in range(len(self.M))])

        # Normaliser pour éviter l'underflow numérique sur de nombreux tours
        b_max = beliefs.max()
        if b_max > 0:
            beliefs = beliefs / b_max
        else:
            beliefs = np.ones(len(self.M))

        for j in range(len(self.M)):
            self.belief_over_M[tuple(self.M[j])] = float(beliefs[j])

        # Dénominateur du belief conjoint: sum_mL B(mL) * compat(mS, mL)
        compat_mL  = (self.M[:, y_opt] <= tau_L).astype(float)  # (|M|,)
        m_S_compat = float(m_S[y_opt] <= tau_S)
        denominator = float((beliefs * compat_mL).sum()) * m_S_compat

        if denominator == 0:
            raise RuntimeError("joint_belief denominator is zero")

        # L0 vectorisé et mis en cache pour ce tour
        L0 = self._l0_matrix(tau_S, tau_L, U_arr, y_opt, w)  # (|M|, |U|)

        # joint_beliefs[j] = B(mL=M[j]) * compat(mS, M[j]) / den
        joint_beliefs = beliefs * compat_mL * m_S_compat / denominator  # (|M|,)

        # scores[k] = sum_j joint_beliefs[j] * log(L0[j, k] + dummy)
        dummy = 1e-12
        scores_vec = joint_beliefs @ np.log(L0 + dummy)  # (|U|,)

        # Filtre lexique: le speaker ne propose que des actions dans son propre lexique
        scores_before_lex = scores_vec.copy()
        lex_speaker = np.array([float(lexicon_with_history(m_S, tau_S, u, w)) for u in U_arr])
        scores_vec = np.where(lex_speaker > 0, scores_vec, -np.inf)

        valid_mask = np.isfinite(scores_vec)
        if not valid_mask.any():
            # L'historique a épuisé toutes les options: relâcher la contrainte d'historique
            lex_base = (np.array([m_S[u] for u in U_arr]) <= tau_S).astype(float)
            scores_vec = np.where(lex_base > 0, scores_before_lex, -np.inf)
            valid_mask = np.isfinite(scores_vec)
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
