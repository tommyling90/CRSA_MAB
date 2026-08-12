import numpy as np

class NegotiationProtocol:
    def __init__(self, game, agent_A, agent_B, algo, U_space, max_turns):
        self.game = game
        self.agents = { "A": agent_A, "B": agent_B }
        self.algo = algo
        self.U_space = U_space
        self.max_turns = max_turns
        self.turn = 0
        self.history = []
        self.listener_y_history = []

    def get_roles(self):
        if self.turn % 2 == 0:
            return "A", "B"  # speaker, listener
        else:
            return "B", "A"

    def run(self):
        final_u = None
        agreement = False
        agreement_turns = None

        while self.turn < self.max_turns:
            speaker_id, listener_id = self.get_roles()
            speaker = self.agents[speaker_id]
            listener = self.agents[listener_id]

            # ====================================================
            # 1. SPEAKER CHOOSES UTTERANCE
            # ====================================================

            utterance = self.algo.choose_utterance(
                speaker,
                listener,
                self.game,
                self.U_space,
                self.turn,
                self.history,
            )
            if utterance is None:
                print({
                    "turn": self.turn,
                    "speaker": speaker_id,
                    "listener": listener_id,
                    "status": "no valid utterances left",
                })

                break

            print({
                "turn": self.turn,
                "utterance": utterance,
                "speaker": speaker_id,
                "listener": listener_id,
            })

            # ====================================================
            # 2. ADD CURRENT UTTERANCE TO HISTORY
            # ====================================================

            self.history.append({
                "turn": self.turn,
                "speaker": speaker_id,
                "listener": listener_id,
                "utterance": utterance,
            })

            # ====================================================
            # 3. CRSA DIAGNOSTICS
            # ====================================================

            if self.algo.name == "crsa":
                # -----------------------------------------------
                # Record belief for existing belief plots
                # -----------------------------------------------

                self.algo.record_beliefs(
                    turn=self.turn,
                    w=self.history,
                    curr_agent=speaker_id,
                    U_space=self.U_space,
                )

                # -----------------------------------------------
                # Cache the REAL CRSA speaker matrix
                # -----------------------------------------------

                self.algo.cache_final_speaker_matrix(
                    speaker.agent_id,
                    listener.agent_id,
                    speaker.tau,
                    listener.tau,
                    self.U_space,
                    self.game.Y_space,
                    self.game.y_opts,
                    self.turn,
                    self.history,
                )

                # -----------------------------------------------
                # Listener distribution over ALL possible y
                #
                # IMPORTANT:
                # prag_listener_y_dist does NOT use the true
                # y_opts to construct the prediction.
                # -----------------------------------------------

                y_dist = self.algo.prag_listener_y_dist(
                    turn=self.turn,
                    speaker_agent=speaker.agent_id,
                    listener_true_meaning=listener.true_meaning,
                    tau_S=speaker.tau,
                    tau_L=listener.tau,
                    U_space=self.U_space,
                    Y_space=self.game.Y_space,
                    utterance=utterance,
                    w=self.history,
                )

                # -----------------------------------------------
                # Find maximum-probability y(s)
                # -----------------------------------------------

                max_prob = max(y_dist.values())

                top_ys = [
                    y
                    for y, prob in y_dist.items()
                    if np.isclose(prob, max_prob)
                ]

                # Keep one representative prediction for logging.
                predicted_y = top_ys[0]

                # -----------------------------------------------
                # Listener accuracy
                #
                # Correct if ANY maximum-probability prediction
                # belongs to the true optimal set Y*.
                # -----------------------------------------------

                correct = any(
                    y in self.game.y_opts
                    for y in top_ys
                )

                self.listener_y_history.append({
                    "turn": self.turn,
                    "listener": listener_id,
                    "utterance": utterance,
                    "y_dist": y_dist.copy(),
                    "predicted_y": predicted_y,
                    "top_ys": list(top_ys),
                    "y_opts": list(self.game.y_opts),
                    "correct": correct,
                })

            # ====================================================
            # 4. CHECK FOR AGREEMENT
            # ====================================================

            if len(self.history) >= 2:

                prev_u = self.history[-2]["utterance"]

                if (
                        utterance is not None
                        and prev_u is not None
                        and utterance == prev_u
                ):
                    agreement = True
                    final_u = int(utterance)

                    # self.turn is zero-indexed:
                    #
                    # turn 0 = first turn
                    # turn 1 = second turn
                    # etc.
                    agreement_turns = self.turn + 1

                    break

            # ====================================================
            # 5. NEXT TURN
            # ====================================================

            self.turn += 1

        # ========================================================
        # NEGOTIATION FINISHED
        # ========================================================

        if agreement:
            turns = agreement_turns
        else:
            turns = self.max_turns

        print({
            "final joint action": final_u,
            "agreement_turns": agreement_turns,
        })

        return final_u, turns, agreement