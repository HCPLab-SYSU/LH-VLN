import numpy as np

class RandomAgent:
    def __init__(self):
        self.actions = ["move_forward", "turn_left", "turn_right", "stop"]
    
    def nav(self, info):
        action = np.random.choice(self.actions)
        return action