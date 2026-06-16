class NegotiationProtocol():
    def __init__(self, agent_A, agent_B, turns, recursion_depth):
        self.agent_A = agent_A
        self.agent_B = agent_B
        self.turns = turns
        self.recursion_depth = recursion_depth

    # run interaction loop (propose -> reject -> propose...)
    # decides when negotiation ends
    # calls actionA.choose_action() e.g.