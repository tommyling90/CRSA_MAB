import numpy as np
from src.rewards.reward_func import calc_reward, penalty

class MatrixGame:
    def __init__(self, payoff_A, payoff_B, Y_space, y_opts, reward_type, episodes):
        self.payoff_A = payoff_A.flatten()
        self.payoff_B = payoff_B.flatten()
        self.Y_space = Y_space
        self.y_opts = list(y_opts)
        self.reward_type = reward_type
        self.episodes = episodes
        self.regret = []
        self.reward = []
        self.episode = 0

    def compute_reward(self, action):
        return calc_reward(self.reward_type, self.payoff_A, self.payoff_B, action)

    def update_step(self, action, turns):
        cost = penalty(turns)
        reward = self.compute_reward(action) - cost if action is not None else -cost
        self.reward.append(reward)
        optimal_reward = max(
            self.payoff_A[y] + self.payoff_B[y]
            for y in self.y_opts
        )

        self.regret.append(
            optimal_reward - reward
        )
        self.episode += 1
        if self.episode == self.episodes:
            print(f"\n=== EPISODES COMPLETED. GAME RESULT ===")
            print("\nrewards:", self.reward)
            print("\nregrets:", self.regret)
            print("\ncumulative regrets:", np.cumsum(self.regret))