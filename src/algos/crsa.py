import numpy as np
from src.priors.lexicon import lexicon_matrix

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
        self.belief_history = []
        self.speaker_history = []

    @staticmethod
    def _optimal_indices(y_opts):
        return np.asarray(y_opts, dtype=int)

    @staticmethod
    def _compatible_with_any_optimal(
            m_S,
            M_L,
            tau_S,
            tau_L,
            y_opts,
    ):
        """
        For one speaker meaning m_S and all listener meanings M_L,
        return how many y* are jointly acceptable.
        Shape: (|M_L|,)
        """
        y = np.asarray(y_opts, dtype=int)

        speaker_ok = (
                np.asarray(m_S)[y] <= tau_S
        )  # (|Y*|,)

        listener_ok = (
                np.asarray(M_L)[:, y] <= tau_L
        )  # (|M_L|, |Y*|)

        return (
                listener_ok
                & speaker_ok[None, :]
        ).sum(axis=1).astype(float)

    def choose_utterance(self, speaker, listener, game, U_space, turn, history):
        M_L = np.array(self.meaning_spaces[listener.agent_id])
        M_S = np.array(self.meaning_spaces[speaker.agent_id])

        dist = self.get_speaker_dist(
            speaker.true_meaning,
            speaker.tau,
            listener.tau,
            U_space,
            game.Y_space,
            game.y_opts,
            turn,
            speaker.agent_id,
            history,
            M_L,
            M_S,
            self.recursion_depth,
            self.alpha
        )
        if dist is None:
            return None

        u = int(
            np.random.choice(
                list(dist.keys()),
                p=list(dist.values())
            )
        )

        self.speaker_history.append({
            "turn": turn,
            "agent": speaker.agent_id,
            "dist": dict(dist),
            "chosen_u": u,
        })

        return u

    #Notez que dans cette matrice c'est "si y_opt a la prob de 1 ou 0 etant donné un certain m_L et u".
    #les autres y ne sont pas pertinents pcq c'est forcément 0 (selon les formules de prior et lexicon)
    def _l0_matrix(self, M_S, M_L, tau_S, tau_L, U_arr, y_opts, w, beliefs_S):
        utterances = tuple(int(event["utterance"]) for event in w)

        cache_key = (
            id(M_S),
            id(M_L),
            tau_S,
            tau_L,
            tuple(U_arr),
            tuple(y_opts),
            utterances,
            tuple(np.round(beliefs_S, 12)),
        )
        if cache_key in self._l0_cache:
            return self._l0_cache[cache_key]

        # calculating the l0 by decomposing the equation into vectorizable components and multiplying them
        y = np.asarray(y_opts, dtype=int)

        speaker_ok = (
                M_S[:, y] <= tau_S
        )  # (|M_S|, |Y*|)

        listener_ok = (
                M_L[:, y] <= tau_L
        )  # (|M_L|, |Y*|)

        lex_S = lexicon_matrix(
            meanings=M_S,
            utterances=U_arr,
            tau=tau_S,
            history=w,
        )

        beliefs_S = np.asarray(
            beliefs_S,
            dtype=float,
        )

        # For every y*, integrate over compatible speaker meanings.
        #
        # shape:
        #   numerator_by_y = (|Y*|, |U|)
        numerator_by_y = (
                (
                        beliefs_S[:, None]
                        * speaker_ok.astype(float)
                ).T
                @ lex_S
        )

        # Total speaker belief mass supporting each y*
        #
        # shape: (|Y*|,)
        denominator_by_y = (
                beliefs_S[:, None]
                * speaker_ok.astype(float)
        ).sum(axis=0)

        # Listener meaning m_L receives contributions only from
        # optimal y* that m_L itself considers acceptable.
        #
        # shape: (|M_L|, |U|)
        numerator = (
                listener_ok.astype(float)
                @ numerator_by_y
        )

        # shape: (|M_L|,)
        denominator = (
                listener_ok.astype(float)
                @ denominator_by_y
        )

        L0 = np.zeros_like(
            numerator,
            dtype=float,
        )

        np.divide(
            numerator,
            denominator[:, None],
            out=L0,
            where=denominator[:, None] > 0,
        )

        self._l0_cache[cache_key] = L0
        return L0

    def get_speaker_dist(self, m_S, tau_S, tau_L, U_space, Y_space, y_opts, turn, curr_agent, w, M_L, M_S, depth, alpha):
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
        compat_counts = self._compatible_with_any_optimal(
            m_S=m_S,
            M_L=M_L,
            tau_S=tau_S,
            tau_L=tau_L,
            y_opts=y_opts,
        )

        denominator = float(
            (beliefs * compat_counts).sum()
        )

        if denominator == 0:
            raise RuntimeError(
                f"joint_belief denominator is zero\n"
                f"depth={depth}, turn={turn}, "
                f"curr_agent={curr_agent}"
            )

        joint_beliefs = (
                beliefs
                * compat_counts
                / denominator
        )

        listener_agent = "B" if curr_agent == "A" else "A"

        beliefs_S_for_L0 = self.belief_vector(
            turn=turn,
            w=w,
            curr_agent=listener_agent,
            U_space=U_space,
        )
        # ======= CALCUL DE V (UTILITÉS) =======#
        if depth == 1:
            listener_matrix = self._l0_matrix(M_S, M_L, tau_S, tau_L, U_arr, y_opts, w, beliefs_S_for_L0)
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
                y_opts=y_opts,
                w=w,
                speaker_agent=curr_agent
            )
        # TODO: need to come up with a cost function (should be a part of prior)
        scores_vec = joint_beliefs @ np.log(listener_matrix + 1e-12)

        # Filtre lexique: le speaker ne propose que des actions dans son propre lexique
        speaker_lexicon = lexicon_matrix(
            meanings=np.asarray(m_S)[None, :],
            utterances=U_arr,
            tau=tau_S,
            history=w,
        )[0]

        valid_mask = speaker_lexicon > 0
        scores_vec = np.where(valid_mask, scores_vec, -np.inf)

        if not valid_mask.any():
            return None

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

    def prag_listener_matrix(self, listener_depth, turn, M_S, M_L, tau_S, tau_L, U_space, Y_space, y_opts, w, speaker_agent):
        if listener_depth < 1:
            raise RuntimeError("prag_listener_matrix is only for listener_depth >= 1")

        U_arr = np.array(sorted(U_space))
        listener_agent = "B" if speaker_agent == "A" else "A"

        y = np.asarray(y_opts, dtype=int)

        speaker_ok = (
                M_S[:, y] <= tau_S
        )

        listener_ok = (
                M_L[:, y] <= tau_L
        )

        # Whether each candidate meaning supports at least one globally optimal action.
        compat_mS = speaker_ok.any(
            axis=1
        ).astype(float)

        compat_mL = listener_ok.any(
            axis=1
        ).astype(float)

        beliefs_S = self.belief_vector(turn, w, listener_agent, U_space)

        b_max = beliefs_S.max()
        beliefs_S = beliefs_S / b_max if b_max > 0 else np.ones(len(M_S))

        # this part is for when recursion depth is only 2 - we don't really need the recursion call
        # if more than 2 it goes to the else condition
        if listener_depth == 1:
            beliefs_L = self.belief_vector(turn, w, speaker_agent, U_space)

            bL_max = beliefs_L.max()
            beliefs_L = beliefs_L / bL_max if bL_max > 0 else np.ones(len(M_L))

            L0 = self._l0_matrix(M_S, M_L, tau_S, tau_L, U_arr, y_opts, w, beliefs_S)

            speaker_mass_by_y = (
                    beliefs_S[:, None]
                    * speaker_ok.astype(float)
            ).sum(axis=0)

            listener_opt_mass = (
                    listener_ok.astype(float)
                    @ speaker_mass_by_y
            )

            weights = (
                    beliefs_L
                    * listener_opt_mass
            )

            if weights.sum() == 0:
                raise RuntimeError(
                    "No compatible listener meanings"
                )

            weights = weights / weights.sum()
            base_scores = weights @ np.log(L0 + 1e-12)

            lex_S = lexicon_matrix(
                meanings=M_S,
                utterances=U_arr,
                tau=tau_S,
                history=w,
            )

            valid = lex_S > 0

            scores = np.where(valid, base_scores[None, :], -np.inf)

            max_scores = np.max(scores, axis=1, keepdims=True)
            exp_scores = np.where(
                np.isfinite(scores),
                np.exp(self.alpha * (scores - max_scores)),
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
                    y_opts=y_opts,
                    turn=turn,
                    curr_agent=speaker_agent,
                    w=w,
                    M_L=M_L,
                    M_S=M_S,
                    depth=listener_depth,
                    alpha=self.alpha
                )
                if S_dist is None:
                    continue

                speaker_matrix[j, :] = np.array([
                    S_dist[int(u)] for u in U_arr
                ])

        # Score each utterance separately for each optimal y*.
        #
        # shape: (|Y*|, |U|)
        score_by_y = np.stack([
            (
                    beliefs_S
                    * speaker_ok[:, j].astype(float)
            ) @ speaker_matrix
            for j in range(len(y_opts))
        ])

        # Each listener meaning only receives score from
        # y* values that it accepts.
        #
        # shape: (|M_L|, |U|)
        numerator = (
                listener_ok.astype(float)
                @ score_by_y
        )
        listener_matrix = np.where(numerator > 0, 1.0, 0.0)

        return listener_matrix

    def prag_listener_y_dist(
            self,
            turn,
            speaker_agent,
            listener_true_meaning,
            tau_S,
            tau_L,
            U_space,
            Y_space,
            utterance,
            w,
    ):
        """
        Diagnostic listener distribution over ALL y in Y_space.

        IMPORTANT:
        The true y_opts are NOT used to construct this distribution.
        They are used only afterward to evaluate accuracy.
        """

        U_arr = np.array(sorted(U_space))
        Y_arr = np.array(sorted(Y_space))

        listener_agent = (
            "B" if speaker_agent == "A" else "A"
        )

        M_S = np.array(
            self.meaning_spaces[speaker_agent]
        )

        # Listener belief over possible speaker meanings
        beliefs_S = self.belief_vector(
            turn=turn,
            w=w,
            curr_agent=listener_agent,
            U_space=U_space,
        )

        total = beliefs_S.sum()

        if total > 0:
            beliefs_S = beliefs_S / total
        else:
            beliefs_S = (
                    np.ones(len(M_S))
                    / len(M_S)
            )

        u_index = {
            int(u): i
            for i, u in enumerate(U_arr)
        }

        u_idx = u_index[int(utterance)]

        scores = np.zeros(
            len(Y_arr),
            dtype=float,
        )

        current_key = (
            turn,
            speaker_agent,
        )

        # Save current REAL CRSA caches because we will temporarily
        # evaluate hypothetical target y values.
        original_current_matrix = (
            self.speaker_matrix_cache.get(current_key)
        )

        original_speaker_cache = (
            self.speaker_cache.copy()
        )

        try:
            # Evaluate every possible y separately
            for j, y in enumerate(Y_arr):

                # Listener knows its own true meaning.
                # If listener would never accept y, its probability is 0.
                if listener_true_meaning[y] > tau_L:
                    continue

                # Remove only the current-turn cached matrix.
                # Previous turns must remain because belief_vector()
                # needs them for history.
                self.speaker_matrix_cache.pop(
                    current_key,
                    None,
                )

                # get_speaker_dist cache does not include y_opts,
                # so clear it while evaluating hypothetical y.
                self.speaker_cache.clear()

                # Hypothesis: THIS y is the target.
                self.cache_final_speaker_matrix(
                    agent_id=speaker_agent,
                    opponent_id=listener_agent,
                    tau_S=tau_S,
                    tau_L=tau_L,
                    U_space=U_space,
                    Y_space=Y_space,
                    y_opts=[int(y)],
                    turn=turn,
                    w=w,
                )

                hypothetical_speaker_matrix = (
                    self.speaker_matrix_cache[
                        current_key
                    ]
                )

                # Likelihood of the actually observed utterance
                likelihood_u = (
                    hypothetical_speaker_matrix[
                    :,
                    u_idx,
                    ]
                )

                # Which possible speaker meanings would accept y?
                speaker_ok = (
                        M_S[:, y] <= tau_S
                ).astype(float)

                scores[j] = np.sum(
                    beliefs_S
                    * likelihood_u
                    * speaker_ok
                )

        finally:
            # Restore REAL CRSA caches
            self.speaker_cache.clear()
            self.speaker_cache.update(
                original_speaker_cache
            )

            if original_current_matrix is not None:
                self.speaker_matrix_cache[
                    current_key
                ] = original_current_matrix
            else:
                self.speaker_matrix_cache.pop(
                    current_key,
                    None,
                )

        Z = scores.sum()

        if Z > 0:
            probs = scores / Z
        else:
            probs = np.zeros_like(scores)

        return {
            int(y): float(probs[j])
            for j, y in enumerate(Y_arr)
        }

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

    def cache_final_speaker_matrix(self, agent_id, opponent_id, tau_S, tau_L, U_space, Y_space, y_opts, turn, w):
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

        y = np.asarray(y_opts, dtype=int)

        # shape: (|M_S|, |Y*|)
        speaker_ok = (
                M_S[:, y] <= tau_S
        ).astype(float)

        # shape: (|M_L|, |Y*|)
        listener_ok = (
                M_L[:, y] <= tau_L
        ).astype(float)

        # For each y*, how much listener-belief mass supports it?
        # shape: (|Y*|,)
        listener_mass_by_y = (
                beliefs[:, None]
                * listener_ok
        ).sum(axis=0)

        # For each candidate speaker meaning m_S:
        # total compatible listener-belief mass over all y*
        #
        # shape: (|M_S|,)
        denominator = (
                speaker_ok
                @ listener_mass_by_y
        )

        if not np.any(denominator > 0):
            raise RuntimeError(
                "No compatible listener meanings"
            )

        beliefs_S_for_L0 = self.belief_vector(
            turn=turn,
            w=w,
            curr_agent=opponent_id,
            U_space=U_space,
        )

        # build L once
        if self.recursion_depth == 1:
            listener_matrix = self._l0_matrix(
                M_S, M_L, tau_S, tau_L, U_arr, y_opts, w[:turn], beliefs_S_for_L0
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
                y_opts=y_opts,
                w=w[:turn],
                speaker_agent=agent_id
            )
        # shape: (|M_L|, |U|)

        log_listener = np.log(
            listener_matrix + 1e-12
        )

        # For each y*, compute the expected listener score
        # under listener meanings compatible with that y*.
        #
        # shape: (|Y*|, |U|)
        score_by_y = np.stack([
            (
                    beliefs
                    * listener_ok[:, j]
            ) @ log_listener
            for j in range(len(y_opts))
        ])

        # Each candidate speaker meaning only gets contribution
        # from y* values that it itself accepts.
        #
        # shape: (|M_S|, |U|)
        score_numerator = (
                speaker_ok
                @ score_by_y
        )

        scores = np.zeros_like(
            score_numerator
        )

        np.divide(
            score_numerator,
            denominator[:, None],
            out=scores,
            where=denominator[:, None] > 0,
        )
        speaker_lexicon = lexicon_matrix(
            meanings=M_S,
            utterances=U_arr,
            tau=tau_S,
            history=w[:turn],
        )

        valid = (
                (speaker_lexicon > 0)
                & (denominator[:, None] > 0)
        )

        scores = np.where(valid, scores, -np.inf)

        row_has_valid = np.any(np.isfinite(scores), axis=1, keepdims=True)

        max_scores = np.where(
            row_has_valid,
            np.max(scores, axis=1, keepdims=True),
            0.0
        )

        exp_scores = np.where(
            np.isfinite(scores),
            np.exp(self.alpha * (scores - max_scores)),
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

    def record_beliefs(self, turn, w, curr_agent, U_space):
        beliefs = self.belief_vector(
            turn=turn,
            w=w,
            curr_agent=curr_agent,
            U_space=U_space,
        )

        total = beliefs.sum()

        if total > 0:
            probs = beliefs / total
        else:
            probs = np.ones(len(beliefs)) / len(beliefs)

        self.belief_history.append({
            "turn": turn,
            "agent": curr_agent,
            "probs": probs.copy(),
        })