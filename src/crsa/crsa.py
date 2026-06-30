import numpy as np

class CRSA:
    def __init__(self, recursion_depth, meaning_spaces):
        self.recursion_depth = recursion_depth
        self.meaning_spaces = meaning_spaces
        self.speaker_cache = {}
        self.listener_cache = {}
        self.meaning_to_idx = {
            agent_id: {
                tuple(m): i
                for i, m in enumerate(np.array(M))
            }
            for agent_id, M in meaning_spaces.items()
        }
        self._l0_cache = {}  # cache: (w_utterances) → (|M|, |U|)

        #for debugging
        self.debug_counts = {
            "S_call": 0,
            "S_cache": 0,
            "L_call": 0,
            "L_cache": 0,
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
        print("listener_cache size:", len(self.listener_cache))
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

    def get_prag_listener(self, depth, turn, M_S, M_L, m_L, tau_S, tau_L, U_space, Y_space, y_opt, u, w, speaker_agent):
        listener_agent = "B" if speaker_agent == "A" else "A"

        compat_mS = (M_S[:, y_opt] <= tau_S).astype(float)  # (|M_S|,) bool
        m_L_compat = float(m_L[y_opt] <= tau_L)
        beliefs = np.array([self.belief(M_S[j], turn, w, listener_agent, depth) for j in range(len(M_S))]) # (|M_S|,)
        speaker_probs = np.zeros(len(M_S))

        for j in range(len(M_S)):
            if compat_mS[j] == 0:
                continue

            speaker_probs[j] = self.get_speaker_dist(
                m_S=M_S[j],
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
                depth=depth
            )[u]

        base_score = float((beliefs * compat_mS * speaker_probs).sum()) * m_L_compat

        scores = {
            y: base_score if y == y_opt else 0.0
            for y in Y_space
        }

        Z = sum(scores.values())

        if Z == 0:
            raise RuntimeError(
                f"Pragmatic listener: Z=0\n"
                f"turn={turn}\n"
                f"u={u}\n"
                f"m_L={m_L}\n"
                f"tau_S={tau_S}, tau_L={tau_L}\n"
                f"compat_mL={m_L_compat}\n"
                f"compatible m_S={compat_mS.sum()}"
            )

        return {y: scores[y] / Z for y in Y_space}

    # Gives Y dist for a given u and a given m_L
    def get_listener_dist(self, depth, turn, M_S, M_L, m_L, tau_S, tau_L, U_space, Y_space, y_opt, u, w, speaker_agent):
        self.debug_counts["L_call"] += 1

        key = (depth, turn, speaker_agent, tuple(m_L), u)
        if key in self.listener_cache:
            self.debug_counts["L_cache"] += 1
            return self.listener_cache[key]

        if self.debug_counts["L_call"] % 1000 == 0:
            print("DEBUG L", self.debug_counts)
            print(f"depth={depth}, turn={turn}, speaker_agent={speaker_agent}, u={u}")

        if depth == 0:
            U_arr = np.array(sorted(U_space))
            L0 = self._l0_matrix(M_S, M_L, tau_S, tau_L, U_arr, y_opt, w)  # (|M|, |U|)

            listener_agent = "B" if speaker_agent == "A" else "A"
            m_idx = self.meaning_to_idx[listener_agent][tuple(m_L)]
            u_idx = {u: i for i, u in enumerate(U_arr)}[u]
            prob_y_opt = float(L0[m_idx, u_idx])

            dist = {
                y: prob_y_opt if y == y_opt else 0.0
                for y in Y_space
            }

        else:
            dist = self.get_prag_listener(depth, turn, M_S, M_L, m_L, tau_S, tau_L, U_space, Y_space, y_opt, u, w, speaker_agent)

        self.listener_cache[key] = dist
        return dist

    def get_speaker_dist(self, m_S, tau_S, tau_L, U_space, Y_space, y_opt, turn, curr_agent, w, M_L, M_S, depth, alpha=1.0):
        self.debug_counts["S_call"] += 1

        if depth < 1:
            raise RuntimeError("Speaker depth must be >= 1")

        key = (depth, turn, curr_agent, tuple(m_S))
        if key in self.speaker_cache:
            self.debug_counts["S_cache"] += 1
            return self.speaker_cache[key]

        U_arr = np.array(sorted(U_space))

        if self.debug_counts["S_call"] % 1000 == 0:
            print("DEBUG S", self.debug_counts)
            print(f"depth={depth}, turn={turn}, agent={curr_agent}, |M_L|={len(M_L)}, |U|={len(U_arr)}")

        # ======= CALCUL DE BELIEF ET BELIEF CONJOINT =======#
        # Calcul des beliefs sur tout le meaning space M_L
        beliefs = np.array([self.belief(M_L[j], turn, w, curr_agent, depth) for j in range(len(M_L))])

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

        listener_matrix = np.zeros((len(M_L), len(U_arr)))
        for i, cand_m_L in enumerate(M_L):
            for j, u in enumerate(U_arr):
                L = self.get_listener_dist(
                    depth=depth - 1,
                    turn=turn,
                    M_S=M_S,
                    M_L=M_L,
                    m_L=cand_m_L,
                    tau_S=tau_S,
                    tau_L=tau_L,
                    U_space=U_space,
                    Y_space=Y_space,
                    y_opt=y_opt,
                    u=u,
                    w=w,
                    speaker_agent=curr_agent
                )

                listener_matrix[i, j] = L[y_opt]

        dummy = 1e-12
        # TODO: need to come up with a cost function (should be a part of prior)
        scores_vec = joint_beliefs @ np.log(listener_matrix + dummy)

        # Filtre lexique: le speaker ne propose que des actions dans son propre lexique
        L0 = self._l0_matrix(M_S, M_L, tau_S, tau_L, U_arr, y_opt, w)
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

        self.speaker_cache[key] = u_dist
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

    def belief(self, cand_m_L, turn, w, curr_agent, depth):
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
            key = (depth, i, other_agent, tuple(cand_m_L))
            #TODO: need to find a way to calculate for speaker_cache where key not exists
            if key not in self.speaker_cache:
                prod *= 1
            else:
                prod *= self.speaker_cache[key][u_i]

        return prod