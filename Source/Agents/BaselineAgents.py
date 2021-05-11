import numpy as np

class RandomAgent:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def getSkewAction(self, state):
        return np.random.uniform(self.low, self.high), None

    def inputPostTrade(self):
        pass

class RecycleAgent:
    def __init__(self, offset):
        self.offset = offset

    def getSkewAction(self, state):
        inventory = state[0]

        if inventory == 0:
            skew = 0.0
        elif inventory < 0:
            skew = self.offset
        else:
            skew = self.offset * -1.0

        return skew, None

    def inputPostTrade(self):
        pass



