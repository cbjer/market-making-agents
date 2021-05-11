import numpy as np

class OrderBook:
    def __init__(self):
        self.reset()

    def reset(self):
        self._orders = []
        self._orderIds = []

    def addOrder(self, bid, ask, dealerId):
        self._orders.append((bid, ask))
        self._orderIds.append(dealerId)

    def getHighestBid(self):
        bids, _ = self._getBidsAsks()
        bestBidRow = bids.argmax()
        bestBid = bids[bestBidRow]
        dealerId = self._orderIds[bestBidRow]
        return bestBid, dealerId

    def getLowestOffer(self):
        _, asks = self._getBidsAsks()
        bestAskRow = asks.argmin()
        bestAsk = asks[bestAskRow]
        dealerId = self._orderIds[bestAskRow]
        return bestAsk, dealerId

    def getMidPrice(self):
        bestBid, _ = self.getHighestBid()
        bestAsk, _ = self.getLowestOffer()
        return np.average([bestBid, bestAsk])

    def _getBidsAsks(self):
        self._checkValidOrderbook()
        orders = np.array(self._orders)
        bids = orders[:, 0]
        asks = orders[:, 1]
        return bids, asks

    def _checkValidOrderbook(self):
        if len(self._orderIds) == 0:
            raise IndexError("No orders have been added to the orderbook")



