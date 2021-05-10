from Source.OrderBook import OrderBook
from Source.OrderArrivalProcess import OrderArrivalProcess
from Source.StockPriceProcess import StockPriceProcess

import numpy as np

ARRIVAL_INTENSITY = 0.1
PRICE_DRIFT = 0.0
PRICE_VOL = 0.05 / np.sqrt(252)
INITIAL_PRICE = 100.0

class ExchangeObservation:
    """
    Information received back to dealer
    """
    def __init__(self, mid, clientDirection, winningPrice, tradeWon, episodeFinished):
        self._mid = mid
        self._clientDirection = clientDirection
        self._winningPrice = winningPrice
        self._tradeWon = tradeWon
        self._episodeFinished = episodeFinished

    def getMid(self):
        return self._mid

    def getClientDirection(self):
        return self._clientDirection

    def getWinningPrice(self):
        return self._winningPrice

    def wasTradeWon(self):
        return self._tradeWon

    def hasEpisodeFinished(self):
        return self._episodeFinished

class StockExchange:
    """
    Interface agents submit their prices to and retrieve back information about trade outcome
    """
    def __init__(self, totalSteps):
        self._orderBook = OrderBook()
        self._stockPriceProcess = StockPriceProcess(totalSteps, PRICE_DRIFT, PRICE_VOL, INITIAL_PRICE)
        self._orderArrivalProcess = OrderArrivalProcess(totalSteps, ARRIVAL_INTENSITY)
        self._totalSteps = totalSteps
        self._clearPreviousWonTrade()

    def reset(self):
        self._stepNumber = 0
        self._orderBook.reset()
        self._orderArrivalProcess.reset()
        self._stockPriceProcess.reset()
        self._clearPreviousWonTrade() # {dealerId : {'tradePrice' : tradePrice, 'clientDirection' : clientDirection}}

    def getInitialPrice(self):
        return self._stockPriceProcess.getPrice(stepNumber=0)

    def getPostTradeInformation(self, dealerId) -> ExchangeObservation:

        if dealerId in self._previousWonTrade:
            tradePrice = self._previousWonTrade[dealerId]['tradePrice']
            clientDirection = self._previousWonTrade[dealerId]['clientDirection']
            tradeWon = True
        else:
            tradePrice = None
            clientDirection = None
            tradeWon = False

        episodeDone = not self.isEpisodeLive()

        return ExchangeObservation(self._currentMid, clientDirection, tradePrice, tradeWon, episodeDone)

    def submitDealerOrder(self, bid, ask, dealerId):
        self._orderBook.addOrder(bid, ask, dealerId)

    def step(self):
        """
        Run after dealers have submitted their prices
        :return:
        """
        isNewOrder = self._orderArrivalProcess.checkIfOrderArrived(self._stepNumber)
        self._currentMid = self._stockPriceProcess.getPrice(self._stepNumber)
        self._clearPreviousWonTrade()

        if isNewOrder:
            clientOrder = self._generateNewOrder()
        else:
            self._orderBook.reset()
            self._stepNumber += 1
            return

        if clientOrder == 'clientBuys':
            winningPrice, winningDealerId = self._orderBook.getLowestOffer()
        elif clientOrder == 'clientSells':
            winningPrice, winningDealerId = self._orderBook.getHighestBid()

        self._orderBook.reset()
        self._addPreviousWonTrade(winningDealerId, winningPrice, clientOrder)
        self._stepNumber += 1

        return

    def _generateNewOrder(self):
        coinFlip = np.random.random_integers(0, 2)

        if coinFlip == 1:
            return 'clientBuys'
        return 'clientSells'

    def _clearPreviousWonTrade(self):
        self._previousWonTrade = {}

    def isEpisodeLive(self):
        return self._stepNumber < self._totalSteps

    def _addPreviousWonTrade(self, dealerId, tradePrice, clientDirection):
        self._previousWonTrade = {dealerId : {'tradePrice' : tradePrice, 'clientDirection' : clientDirection}}





