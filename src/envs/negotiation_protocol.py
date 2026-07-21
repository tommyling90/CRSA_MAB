from utils.plots import save_top_belief_plot, save_belief_legend

class NegotiationProtocol:
    def __init__(self, game, agent_A, agent_B, algo, U_space, max_turns):
        self.game = game
        self.agents = { "A": agent_A, "B": agent_B }
        self.algo = algo
        self.U_space = U_space
        self.max_turns = max_turns
        self.turn = 0
        self.history = []

    def get_roles(self):
        if self.turn % 2 == 0:
            return "A", "B"  # speaker, listener
        else:
            return "B", "A"

    def run(self):
        final_u = None
        agreement = False

        while self.turn < self.max_turns:
            speaker_id, listener_id = self.get_roles()
            speaker = self.agents[speaker_id]
            listener = self.agents[listener_id]

            utterance = self.algo.choose_utterance(speaker, listener, self.game, self.U_space, self.turn, self.history)
            print({"turn": self.turn, "utterance": utterance, "speaker": speaker_id, "listener": listener_id})
            self.history.append({
                "turn": self.turn,
                "speaker": speaker_id,
                "listener": listener_id,
                "utterance": utterance
            })

            if self.algo.name == "crsa":
                self.algo.print_top_beliefs(self.turn, self.history, speaker_id, self.U_space, 20)
                rows = self.algo.get_top_beliefs(self.turn, self.history, speaker_id, self.U_space, 20)
                save_top_belief_plot(rows, self.turn, speaker_id)
                save_belief_legend(rows, self.turn, speaker_id)

            if len(self.history) >= 2:
                prev_u = self.history[-2]["utterance"]

                if utterance == prev_u:
                    agreement = True
                    final_u = utterance
                    break

            if self.algo.name == "crsa":
                self.algo.cache_final_speaker_matrix(
                    speaker.agent_id, listener.agent_id, speaker.tau, listener.tau, self.U_space, self.game.Y_space,
                    self.game.y_opt, self.turn, self.history,
                )

            self.turn += 1

        # TODO: when ready to run episodes, uncomment this line to update the game and print
        # self.game.update_step(final_u, self.turn+1)

        print({"final joint action": final_u})
        return final_u, self.turn + 1, agreement