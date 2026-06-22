import numpy as np

class BaseAgent:
    def __init__(self, agent_id, payoff):
        self.agent_id = agent_id
        self.payoff = payoff.flatten()
        self.k = len(self.payoff)
        self.regret = []
        self.reward = []
        self.action_counts = np.zeros(self.k, dtype=int)
        self.mean_rewards = np.zeros(self.k)

    def exec_algo(self):
        return

    def choose_action(self, joint_action, game):
        return

    def update(self, action):
        reward = self.payoff[action] if action is not None else 0
        self.reward.append(reward)
        self.regret.append(max(self.payoff) - reward)

        self.action_counts[action] += 1
        n = self.action_counts[action]
        old_mean = self.mean_rewards[action]
        self.mean_rewards[action] = old_mean + (reward - old_mean) / n