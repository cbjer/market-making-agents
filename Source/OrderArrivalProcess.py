import numpy as np

class OrderArrivalProcess:
    """
    Controls if an order has arrived at a given step
    """
    def __init__(self, totalSteps, arrivalIntensity):
        self._totalSteps = totalSteps
        self._arrivalIntensity = arrivalIntensity
        self.reset()

    def reset(self):
        geometric = np.random.geometric(self._arrivalIntensity, self._totalSteps).cumsum()
        self._arrivals = np.zeros(self._totalSteps)

        for entry in geometric:
            if entry >= self._totalSteps:
                break
            self._arrivals[entry] = 1

        self._arrivals[0] = 0

        return

    def checkIfOrderArrived(self, stepNumber):
        return self._arrivals[stepNumber] == 1



