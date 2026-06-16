class Protocol():
    def __init__(self, agent_A, agent_B):
        self.agent_A = agent_A
        self.agent_B = agent_B

    # run interaction loop (propose -> reject -> propose...)
    # decides when negotiation ends
    # calls actionA.choose_action() e.g.