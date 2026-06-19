class MatrixGame:
    def __init__(self, payoff_A, payoff_B, Y_space, y_opt):
        self.payoff_A = payoff_A.flatten()
        self.payoff_B = payoff_B.flatten()
        self.Y_space = Y_space
        self.y_opt = y_opt
        self.k = payoff_A.shape[0]

    def computer_reward(self):
        return

    def updateStep(self):
        #ONLY FOR AFTER AN AGREEMENT IS REACHED
        #given state + action -> next state + reward + info
        #payoff lookup
        return