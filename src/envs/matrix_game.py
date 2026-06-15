import numpy as np

class MatrixGame:
    def __init__(self, matrices):
        self.matrices = matrices

    def updateStep(self):
        #ONLY FOR AFTER AN AGREEMENT IS REACHED
        #given state + action -> next state + reward + info
        #payoff lookup
        return