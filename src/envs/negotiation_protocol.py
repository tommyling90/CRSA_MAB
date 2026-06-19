class NegotiationProtocol:
    def __init__(self, game, agent_A, agent_B, crsa, U_space, max_turns):
        self.game = game
        self.agents = { "A": agent_A, "B": agent_B }
        self.crsa = crsa
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
        #TODO: verify
        listener = utterance = None
        while self.turn < self.max_turns:
            speaker_id, listener_id = self.get_roles()
            speaker = self.agents[speaker_id]
            listener = self.agents[listener_id]

            utterance = self.crsa.choose_utterance(speaker, listener, self.game, self.U_space, self.turn, self.history)

            self.history.append({
                "turn": self.turn,
                "speaker": speaker_id,
                "listener": listener_id,
                "utterance": utterance
            })

            if utterance == "accept":
                break

            self.turn += 1

        if listener and utterance == "accept":
            listener.choose_action(self.history[-2]["utterance"], self.game)
        else:
            raise RuntimeError("No agreement was reached!")

    # run interaction loop (propose -> reject -> propose...)
    # decides when negotiation ends
    # calls actionA.choose_action() e.g.