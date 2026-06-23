from src.rewards.reward_func import calc_reward, penalty

class MatrixGame:
    def __init__(self, payoff_A, payoff_B, Y_space, y_opt, reward_type):
        self.payoff_A = payoff_A.flatten()
        self.payoff_B = payoff_B.flatten()
        self.Y_space = Y_space
        self.y_opt = y_opt
        self.reward_type = reward_type
        self.regret = []
        self.reward = []

    def compute_reward(self, action):
        return calc_reward(self.reward_type, self.payoff_A, self.payoff_B, action)

    def update_step(self, action, turns):
        cost = penalty(turns)
        reward = self.compute_reward(action) - cost if action is not None else -cost
        self.reward.append(reward)
        self.regret.append(self.payoff_A[self.y_opt] + self.payoff_B[self.y_opt] - reward)