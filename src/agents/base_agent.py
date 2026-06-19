class BaseAgent:
    def __init__(self, agent_id, payoff):
        self.agent_id = agent_id
        self.payoff = payoff.flatten()

    def exec_algo(self):
        return

    def choose_action(self, joint_action, game):
        return

    def update(self, reward):
        return