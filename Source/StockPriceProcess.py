import numpy as np

class StockPriceProcess:
    """
    Creates a random walk based stock price
    """
    def __init__(self, totalSteps, drift, vol, initialPrice):
        self._totalSteps = totalSteps
        self._drift = drift
        self._vol = vol
        self._initialPrice = initialPrice
        self.reset()

    def reset(self):
        stockReturns = np.random.normal(self._drift, self._vol, self._totalSteps) + 1.0
        stockMultipliers = stockReturns.cumprod()
        self._priceProcess = self._initialPrice * stockMultipliers

    def getPrice(self, stepNumber):
        return self._priceProcess[stepNumber]
