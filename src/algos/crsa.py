import numpy as np

class CRSA:
    def __init__(self, recursion_depth, meaning_spaces, taus, alpha):
        self.recursion_depth = recursion_depth
        self.meaning_spaces = meaning_spaces
        self.taus = taus
        self.alpha = alpha
        self.speaker_cache = {}
        self._l0_cache = {}  # cache: (w_utterances) → (|M|, |U|)
        self.speaker_matrix_cache = {}
        self.name = "crsa"

    def choose_utterance(self, speaker, listener, game, U_space, turn, history):
        M_L = np.array(self.meaning_spaces[listener.agent_id])
        M_S = np.array(self.meaning_spaces[speaker.agent_id])
        dist = self.get_speaker_dist(speaker.true_meaning, speaker.tau, listener.tau, U_space, game.Y_space, game.y_opt, turn, speaker.agent_id, history, M_L, M_S, self.recursion_depth, self.alpha)
        # self.print_speaker_u_dist(dist, speaker.agent_id, turn)
        u = np.random.choice(list(dist.keys()), p=list(dist.values()))
        return u

    #Notez que dans cette matrice c'est "si y_opt a la prob de 1 ou 0 etant donné un certain m_L et u".
    #les autres y ne sont pas pertinents pcq c'est forcément 0 (selon les formules de prior et lexicon)
    def _l0_matrix(self, M_S, M_L, tau_S, tau_L, U_arr, y_opt, w):
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

    def get_speaker_dist(self, m_S, tau_S, tau_L, U_space, Y_space, y_opt, turn, curr_agent, w, M_L, M_S, depth, alpha):
        if depth < 1:
            raise RuntimeError("Speaker depth must be >= 1")

        key = (depth, turn, curr_agent, tuple(m_S))
        if key in self.speaker_cache:
            return self.speaker_cache[key]

        U_arr = np.array(sorted(U_space))

        # ======= CALCUL DE BELIEF ET BELIEF CONJOINT =======#
        beliefs = self.belief_vector(turn, w, curr_agent, U_space)
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
        if listener_depth < 1:
            raise RuntimeError("prag_listener_matrix is only for listener_depth >= 1")

        U_arr = np.array(sorted(U_space))
        listener_agent = "B" if speaker_agent == "A" else "A"

        compat_mS = (M_S[:, y_opt] <= tau_S).astype(float)  # (|M_S|,)
        compat_mL = (M_L[:, y_opt] <= tau_L).astype(float)  # (|M_L|,)

        beliefs_S = self.belief_vector(turn, w, listener_agent, U_space)

        b_max = beliefs_S.max()
        beliefs_S = beliefs_S / b_max if b_max > 0 else np.ones(len(M_S))

        # this part is for when recursion depth is only 2 - we don't really need the recursion call
        # if more than 2 it goes to the else condition
        if listener_depth == 1:
            beliefs_L = self.belief_vector(turn, w, speaker_agent, U_space)

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
                    depth=listener_depth,
                    alpha=self.alpha
                )

                speaker_matrix[j, :] = np.array([
                    S_dist[int(u)] for u in U_arr
                ])

        score_u = (beliefs_S * compat_mS) @ speaker_matrix  # (|U|,)
        numerator = compat_mL[:, None] * score_u[None, :]  # (|M_L|, |U|)
        listener_matrix = np.where(numerator > 0, 1.0, 0.0)

        return listener_matrix

    def belief_vector(self, turn, w, curr_agent, U_space):
        other_agent = "B" if curr_agent == "A" else "A"
        M_other = np.array(self.meaning_spaces[other_agent])
        beliefs = np.ones(len(M_other))

        if turn == 0:
            return beliefs

        U_arr = np.array(sorted(U_space))
        u_index = {int(u): idx for idx, u in enumerate(U_arr)}

        for i, event in enumerate(w[:turn]):
            if event["speaker"] != other_agent:
                continue

            key = (i, other_agent)
            speaker_matrix = self.speaker_matrix_cache[key]

            u_idx = u_index[int(event["utterance"])]

            beliefs *= speaker_matrix[:, u_idx]

        return beliefs

    def cache_final_speaker_matrix(self, agent_id, opponent_id, tau_S, tau_L, U_space, Y_space, y_opt, turn, w):
        key = (turn, agent_id)

        if key in self.speaker_matrix_cache:
            return

        U_arr = np.array(sorted(U_space))
        M_S = np.array(self.meaning_spaces[agent_id])
        M_L = np.array(self.meaning_spaces[opponent_id])

        if turn == 0:
            beliefs = np.ones(len(M_L))
        else:
            beliefs = self.belief_vector(turn, w, agent_id, U_space)

        b_max = beliefs.max()
        beliefs = beliefs / b_max if b_max > 0 else np.ones(len(M_L))

        compat_mS = (M_S[:, y_opt] <= tau_S).astype(float)  # (|M_S|)
        compat_mL = (M_L[:, y_opt] <= tau_L).astype(float)  # (|M_L|)

        # denominator per m_S
        denom_base = (beliefs * compat_mL).sum()

        if denom_base == 0:
            raise RuntimeError("No compatible listener meanings")

        # joint beliefs for every m_S
        # rows for incompatible m_S will be zero
        joint = compat_mS[:, None] * (beliefs * compat_mL)[None, :] / denom_base
        # shape: (|M_S|, |M_L|)

        # build L once
        if self.recursion_depth == 1:
            listener_matrix = self._l0_matrix(
                M_S, M_L, tau_S, tau_L, U_arr, y_opt, w[:turn]
            )
        else:
            listener_matrix = self.prag_listener_matrix(
                listener_depth=self.recursion_depth - 1,
                turn=turn,
                M_S=M_S,
                M_L=M_L,
                tau_S=tau_S,
                tau_L=tau_L,
                U_space=U_space,
                Y_space=Y_space,
                y_opt=y_opt,
                w=w[:turn],
                speaker_agent=agent_id
            )
        # shape: (|M_L|, |U|)

        scores = joint @ np.log(listener_matrix + 1e-12)
        # shape: (|M_S|, |U|)

        utterances = [event["utterance"] for event in w[:turn]]
        last_u = utterances[-1] if utterances else None

        hist_mask = np.array([
            not (u in utterances and u != last_u)
            for u in U_arr
        ])

        lex_mask = (M_S[:, U_arr] <= tau_S)
        valid = lex_mask & hist_mask[None, :] & (compat_mS[:, None] == 1)

        scores = np.where(valid, scores, -np.inf)

        row_has_valid = np.any(np.isfinite(scores), axis=1, keepdims=True)

        max_scores = np.where(
            row_has_valid,
            np.max(scores, axis=1, keepdims=True),
            0.0
        )

        exp_scores = np.where(
            np.isfinite(scores),
            np.exp(scores - max_scores),
            0.0
        )

        Z = exp_scores.sum(axis=1, keepdims=True)

        speaker_matrix = np.zeros_like(exp_scores)
        np.divide(exp_scores, Z, out=speaker_matrix, where=(Z > 0))

        self.speaker_matrix_cache[key] = speaker_matrix

    def print_top_beliefs(self, turn, w, curr_agent, U_space, top_k=20):
        other_agent = "B" if curr_agent == "A" else "A"
        rows = self.get_top_beliefs(turn, w, curr_agent, U_space, top_k)

        print(f"\n=== Turn {turn}: {curr_agent}'s top {top_k} beliefs over {other_agent}'s meanings ===")

        for row in rows:
            print(
                f"{row['rank']:02d}. "
                f"prob={row['prob']:.6f}, "
                f"meaning={row['meaning']}"
            )

    def get_top_beliefs(self, turn, w, curr_agent, U_space, top_k=20):
        other_agent = "B" if curr_agent == "A" else "A"
        M_other = np.array(self.meaning_spaces[other_agent])

        beliefs = self.belief_vector(
            turn=turn,
            w=w,
            curr_agent=curr_agent,
            U_space=U_space
        )

        total = beliefs.sum()
        probs = beliefs / total if total > 0 else np.ones(len(beliefs)) / len(beliefs)

        top_idx = np.argsort(probs)[-top_k:][::-1]

        rows = []
        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                "rank": rank,
                "meaning_id": f"m_{idx}",
                "meaning": M_other[idx].tolist(),
                "prob": float(probs[idx])
            })

        return rows

    def print_speaker_u_dist(self, dist, speaker_id, turn):
        items = sorted(dist.items(), key=lambda x: x[1], reverse=True)

        print(f"\n=== Turn {turn}: Speaker {speaker_id}'s distribution over utterances ===")

        for u, p in items:
            print(f"u={u}, prob={p:.6f}")