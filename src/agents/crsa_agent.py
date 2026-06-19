from src.agents.base_agent import BaseAgent

class CRSAAgent(BaseAgent):
    def __init__(self, agent_id, payoff, true_meaning, tau):
        super().__init__(agent_id, payoff)
        self.true_meaning = true_meaning
        self.tau = tau

    def choose_action(self, joint_action, game):
        reward = game.payoff_A[joint_action] + game.payoff_B[joint_action]
        return

    def update(self, reward):
        #update arm counts, means,
        return