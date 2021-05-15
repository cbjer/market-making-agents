import numpy as np

class RandomAgent:
    def __init__(self, low, high, name):
        self.low = low
        self.high = high

    def getSkewAction(self, state):
        return np.random.uniform(self.low, self.high), None

    def inputPostTrade(self, state, action, reward, done, nextState):
        return

class RecycleAgent:
    def __init__(self, offset, name):
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

    def inputPostTrade(self, state, action, reward, done, nextState):
        return

class ImprovedRecycleAgent:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name

    def getSkewAction(self, state):
        inventory = state.inventory
        skew = (inventory / 10.0) * -1.0
        rnd = np.random.uniform(-0.05, 0.05)
        return np.max([-1.0, np.min([skew, 1.0])]) + rnd

    def inputPostTrade(self, state, action, reward, done, nextState):
        return





