import numpy as np

class RandomAgent:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def getSkewAction(self, state):
        return np.random.uniform(self.low, self.high), None

class RecycleAgent:
    def __init__(self, offset):
        self.offset = offset

    def getSkewAction(self, state):
        inventory = state.inventory

        if inventory == 0:
            skew = 0.0
        elif inventory < 0:
            # Agent needs to buy -> high bid -> positive skew
            skew = self.offset
        else:
            # Agent needs to sell -> low offer -> negative skew
            skew = self.offset * -1.0

        return skew, None

class ImprovedRecycleAgent:
    def __init__(self):
        pass

    def getSkewAction(self, state):
        inventory = state.inventory
        skew = (inventory / 10.0) * -1.0
        return np.max([-1.0, np.min([skew, 1.0])])




