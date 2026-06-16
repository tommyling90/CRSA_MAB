from base_agent import BaseAgent
from src.crsa.speaker import *
from src.crsa.listener import *
from src.crsa.beliefs import *
from src.crsa.recursion import *

class CRSAAgent(BaseAgent):
    def __init__(self, payoff, true_meaning, meaning_space):
        super().__init__(payoff)
        self.true_meaning = true_meaning
        self.meaning_space = meaning_space

        # u?
        # last_w?

    def update_belief(self, belief):
        return

    def choose_action(self):
        return

    def exec_speaking_algo(self):
        # choose utterance (propose ou accepte)
        return

    def exec_listening_algo(self):
        return

    def exec_algo(self):
        return

    def update(self, reward):
        #update arm counts, means,
        return