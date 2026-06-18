class NegotiationProtocol():
    def __init__(self, agent_A, agent_B, num_turns, recursion_depth):
        self.agents = { "A": agent_A, "B": agent_B }
        self.num_turns = num_turns
        self.recursion_depth = recursion_depth
        self.turn = 0
        #TODO: link up cache with speaker dist/belief
        self.speaker_cache = {}
        #TODO: need to add to history
        self.history = []

    def get_roles(self):
        if self.turn % 2 == 0:
            return "A", "B"  # speaker, listener
        else:
            return "B", "A"

    def exec_turn(self):
        if self.turn < self.num_turns:
            speaker_id, listener_id = self.get_roles()
            speaker = self.agents[speaker_id]
            listener = self.agents[listener_id]

        self.turn += 1
        return

    # run interaction loop (propose -> reject -> propose...)
    # decides when negotiation ends
    # calls actionA.choose_action() e.g.