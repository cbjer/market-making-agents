from Source.OrderBook import OrderBook
from Source.OrderArrivalProcess import OrderArrivalProcess
from Source.StockPriceProcess import StockPriceProcess

import numpy as np
from collections import namedtuple

ARRIVAL_INTENSITY = 0.2
PRICE_DRIFT = 0.0
PRICE_VOL = 0.01 / np.sqrt(252)
INITIAL_PRICE = 100.0

ExchangeObservation = namedtuple("ExchangeObservation", ['mid', 'clientDirection', 'winningPrice', 'tradeWon', 'episodeFinished', 'tradeLost'])

class StockExchange:
    """
    Interface agents submit their prices to and retrieve back information about trade outcome
    """
    def __init__(self, totalSteps):
        self._orderBook = OrderBook()
        self._stockPriceProcess = StockPriceProcess(totalSteps, PRICE_DRIFT, PRICE_VOL, INITIAL_PRICE)
        self._orderArrivalProcess = OrderArrivalProcess(totalSteps, ARRIVAL_INTENSITY)
        self._totalSteps = totalSteps
        self._previousWonTrade = {}

    def reset(self):
        self._stepNumber = 0
        self._orderBook.reset()
        self._orderArrivalProcess.reset()
        self._stockPriceProcess.reset()
        self._previousWonTrade = {} # {dealerId : {'tradePrice' : tradePrice, 'clientDirection' : clientDirection}}

    def getInitialPrice(self):
        return self._stockPriceProcess.getPrice(stepNumber=0)

    def getPostTradeInformation(self, dealerId):

        if dealerId in self._previousWonTrade:
            tradePrice = self._previousWonTrade[dealerId]['tradePrice']
            clientDirection = self._previousWonTrade[dealerId]['clientDirection']
            tradeWon = True
            tradeLost = False

        elif len(self._previousWonTrade) > 0:
            tradePrice = None
            clientDirection = None
            tradeWon = False
            tradeLost = True
        else:
            tradePrice = None
            clientDirection = None
            tradeWon = False
            tradeLost = False


        episodeDone = not self.isEpisodeLive()

        return ExchangeObservation(self._currentMid, clientDirection, tradePrice, tradeWon, episodeDone, tradeLost)

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
            clientOrder = self._generateInformedRandomNewOrder(self._currentMid, self._orderBook.getMidPrice())
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
        coinFlip = np.random.random_integers(0, 1)

        if coinFlip == 1:
            return 'clientBuys'
        return 'clientSells'

    def _generateInformedInvestorNewOrder(self, trueMid, orderBookMid):
        if trueMid > orderBookMid:
            return 'clientBuys'
        return 'clientSells'

    def _generateInformedRandomNewOrder(self, trueMid, orderBookMid):
        coinFlip = np.random.random_integers(0, 4)

        if coinFlip == 0:
            return self._generateInformedInvestorNewOrder(trueMid, orderBookMid)
        else:
            return self._generateNewOrder()

    def _clearPreviousWonTrade(self):
        assert len(self._previousWonTrade) <= 1
        self._previousWonTrade = {}

    def isEpisodeLive(self):
        return self._stepNumber < self._totalSteps

    def _addPreviousWonTrade(self, dealerId, tradePrice, clientDirection):
        self._previousWonTrade = {dealerId : {'tradePrice' : tradePrice, 'clientDirection' : clientDirection}}
