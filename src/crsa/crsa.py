import numpy as np

class CRSA:
    def __init__(self, recursion_depth, meaning_spaces):
        self.recursion_depth = recursion_depth
        self.meaning_spaces = meaning_spaces
        self.speaker_cache = {}
        self.belief_cache = {}
        self._l0_cache = {}  # cache: (w_utterances) → (|M|, |U|)
        self.debug_counts = {
            "S_call": 0,
            "S_cache": 0,
            "L0_call": 0,
            "L0_cache": 0,
        }

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
        dist = self.get_speaker_dist(speaker.true_meaning, speaker.tau, listener.tau, U_space, game.Y_space, game.y_opt, turn, speaker.agent_id, history, M_L, M_S, self.recursion_depth)
        u = np.random.choice(
            list(dist.keys()),
            p=list(dist.values())
        )
        print("FINAL DEBUG COUNTS:", self.debug_counts)
        print("speaker_cache size:", len(self.speaker_cache))
        print("l0_cache size:", len(self._l0_cache))
        return u

    #Notez que dans cette matrice c'est si y_opt a la prob de 1 ou 0 etant donné un certain m_L et u.
    #les autres y ne sont pas pertinents pcq c'est forcément 0 (selon les formules de prior et lexicon)
    def _l0_matrix(self, M_S, M_L, tau_S, tau_L, U_arr, y_opt, w):
        self.debug_counts["L0_call"] += 1

        utterances = [event["utterance"] for event in w]
        last_u = utterances[-1] if utterances else None

        cache_key = (
            id(M_S),
            id(M_L),
            tau_S,
            tau_L,
            tuple(U_arr),
            y_opt,
            tuple(utterances),
        )
        if cache_key in self._l0_cache:
            self.debug_counts["L0_cache"] += 1
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

    def get_speaker_dist(self, m_S, tau_S, tau_L, U_space, Y_space, y_opt, turn, curr_agent, w, M_L, M_S, depth, alpha=1.0):
        self.debug_counts["S_call"] += 1

        if depth < 1:
            raise RuntimeError("Speaker depth must be >= 1")

        key = (depth, turn, curr_agent, tuple(m_S))
        if key in self.speaker_cache:
            self.debug_counts["S_cache"] += 1
            return self.speaker_cache[key]

        U_arr = np.array(sorted(U_space))

        # ======= CALCUL DE BELIEF ET BELIEF CONJOINT =======#
        # Calcul des beliefs sur tout le meaning space M_L
        beliefs = np.array([self.belief(M_L[j], turn, w, curr_agent, depth) for j in range(len(M_L))])

        # Normaliser pour éviter l'underflow numérique sur de nombreux tours
        b_max = beliefs.max()
        beliefs = beliefs / b_max if b_max > 0 else np.ones(len(M_L))

        # Dénominateur du belief conjoint: sum_mL B(mL) * compat(mS, mL)
        compat_mL = (M_L[:, y_opt] <= tau_L).astype(float)  # (|M|,)
        m_S_compat = float(m_S[y_opt] <= tau_S)
        denominator = float((beliefs * compat_mL).sum()) * m_S_compat

        if denominator == 0:
            raise RuntimeError(
                f"joint_belief denominator is zero\n"
                f"depth={depth}, turn={turn}, curr_agent={curr_agent}"
            )

        joint_beliefs = beliefs * compat_mL * m_S_compat / denominator  # (|M_L|,)

        # ======= CALCUL DE V (UTILITÉS) =======#
        if depth == 1:
            listener_matrix = self._l0_matrix(
                M_S, M_L, tau_S, tau_L, U_arr, y_opt, w
            )
        else:
            listener_matrix = self.prag_listener_matrix(
                listener_depth=depth - 1,
                turn=turn,
                M_S=M_S,
                M_L=M_L,
                tau_S=tau_S,
                tau_L=tau_L,
                U_space=U_space,
                Y_space=Y_space,
                y_opt=y_opt,
                w=w,
                speaker_agent=curr_agent
            )
        # TODO: need to come up with a cost function (should be a part of prior)
        scores_vec = joint_beliefs @ np.log(listener_matrix + 1e-12)

        # Filtre lexique: le speaker ne propose que des actions dans son propre lexique
        utterances = [event["utterance"] for event in w]
        last_u = utterances[-1] if utterances else None

        hist_mask = np.array([
            not (u in utterances and u != last_u)
            for u in U_arr
        ])

        valid_mask = (m_S[U_arr] <= tau_S) & hist_mask
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

        self.speaker_cache[key] = u_dist
        return u_dist

    def prag_listener_matrix(self, listener_depth, turn, M_S, M_L, tau_S, tau_L, U_space, Y_space, y_opt, w, speaker_agent):
        """
        Returns matrix L_listener_depth[m_L, u] = P(y_opt | u, m_L, w)

        Assumes listener_depth >= 1.
        depth 0 is handled by _l0_matrix outside this function.
        """
        if listener_depth < 1:
            raise RuntimeError("prag_listener_matrix is only for listener_depth >= 1")

        U_arr = np.array(sorted(U_space))
        listener_agent = "B" if speaker_agent == "A" else "A"

        compat_mS = (M_S[:, y_opt] <= tau_S).astype(float)  # (|M_S|,)
        compat_mL = (M_L[:, y_opt] <= tau_L).astype(float)  # (|M_L|,)

        beliefs_S = np.array([
            self.belief(M_S[j], turn, w, listener_agent, listener_depth)
            for j in range(len(M_S))
        ])

        b_max = beliefs_S.max()
        beliefs_S = beliefs_S / b_max if b_max > 0 else np.ones(len(M_S))

        # this part is for when recursion depth is only 2 - we don't really need the recursion call
        # if more than 2 it goes to the else condition
        if listener_depth == 1:
            # Build all S1(m_S, u) at once.
            beliefs_L = np.array([
                self.belief(M_L[i], turn, w, speaker_agent, depth=1)
                for i in range(len(M_L))
            ])

            bL_max = beliefs_L.max()
            beliefs_L = beliefs_L / bL_max if bL_max > 0 else np.ones(len(M_L))

            L0 = self._l0_matrix(M_S, M_L, tau_S, tau_L, U_arr, y_opt, w)

            weights = beliefs_L * compat_mL
            weights = weights / weights.sum()
            base_scores = weights @ np.log(L0 + 1e-12)

            utterances = [event["utterance"] for event in w]
            last_u = utterances[-1] if utterances else None

            hist_mask = np.array([
                not (u in utterances and u != last_u)
                for u in U_arr
            ])

            lex_S = (M_S[:, U_arr] <= tau_S)
            valid = lex_S & hist_mask[None, :]

            scores = np.where(valid, base_scores[None, :], -np.inf)

            max_scores = np.max(scores, axis=1, keepdims=True)
            exp_scores = np.where(
                np.isfinite(scores),
                np.exp(scores - max_scores),
                0.0
            )

            Z = exp_scores.sum(axis=1, keepdims=True)
            speaker_matrix = np.where(Z > 0, exp_scores / Z, 0.0)
        else:

            speaker_matrix = np.zeros((len(M_S), len(U_arr)))

            for j, cand_m_S in enumerate(M_S):
                if compat_mS[j] == 0:
                    continue

                S_dist = self.get_speaker_dist(
                    m_S=cand_m_S,
                    tau_S=tau_S,
                    tau_L=tau_L,
                    U_space=U_space,
                    Y_space=Y_space,
                    y_opt=y_opt,
                    turn=turn,
                    curr_agent=speaker_agent,
                    w=w,
                    M_L=M_L,
                    M_S=M_S,
                    depth=listener_depth
                )

                speaker_matrix[j, :] = np.array([
                    S_dist[int(u)] for u in U_arr
                ])

        score_u = (beliefs_S * compat_mS) @ speaker_matrix  # (|U|,)
        numerator = compat_mL[:, None] * score_u[None, :]  # (|M_L|, |U|)
        listener_matrix = np.where(numerator > 0, 1.0, 0.0)

        return listener_matrix

    def belief(self, cand_m_L, turn, w, curr_agent, depth):
        belief_key = (depth, turn, curr_agent, tuple(cand_m_L))

        if belief_key in self.belief_cache:
            return self.belief_cache[belief_key]

        if not w:
            self.belief_cache[belief_key] = 1.0
            return 1.0
        prod = 1.0

        other_agent = "B" if curr_agent == "A" else "A"

        for i, event in enumerate(w):
            if i >= turn:
                break
            if event["speaker"] != other_agent:
                continue

            u_i = event["utterance"]
            key = (depth, i, other_agent, tuple(cand_m_L))
            #TODO: need to find a way to calculate for speaker_cache where key not exists
            if key not in self.speaker_cache:
                prod *= 1
            else:
                prod *= self.speaker_cache[key][u_i]
        self.belief_cache[belief_key] = prod
        return prod

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