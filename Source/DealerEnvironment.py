import gym
from gym import spaces
import numpy as np

from collections import namedtuple

MINIMUM_SKEW = -1.0
MAXIMUM_SKEW = 1.0

BID_OFFER_WIDTH = 2.0

MINIMUM_PRICE = 0.0
MINIMUM_INVENTORY = -10
MAXIMUM_INVENTORY = 10

INVENTORY_STEP_SIZE = 1.0

ALPHA_PENALTY = 0.05

Observation = namedtuple('Observation', ['inventory'])

class DealerEnvironment(gym.Env):
    """
    Environment which a single dealer agent interacts with

    action_space : skew value in [-1, 1]
    observation_space : (currentInventory, currentMid, tradeWon)
    """
    def __init__(self, dealerId, exchange):
        super(DealerEnvironment, self).__init__()
        self.action_space = spaces.Box(low=np.array([MINIMUM_SKEW]), high=np.array([MAXIMUM_SKEW]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([MINIMUM_INVENTORY, MINIMUM_PRICE]), high=np.array([MAXIMUM_INVENTORY, np.inf]), dtype=np.float32)
        self._dealerId = dealerId
        self._exchange = exchange

    def submitPricesToExchange(self, bid, ask):
        self._exchange.submitDealerOrder(bid, ask, self._dealerId)

    def submitSkewToExchange(self, skew):
        latestMid = self._previousMid
        bid = latestMid - BID_OFFER_WIDTH * ( 1 - skew ) / 2.0
        ask = bid + BID_OFFER_WIDTH

        # Limit to inventory - Have to be super competitive other side if too high
        if self._inventory >= MAXIMUM_INVENTORY:
            ask = latestMid - BID_OFFER_WIDTH
        if self._inventory <= MINIMUM_INVENTORY:
            bid = latestMid + BID_OFFER_WIDTH

        self.submitPricesToExchange(bid, ask)

    def step(self):
        exchangeObservation = self._exchange.getPostTradeInformation(self._dealerId)

        mid = exchangeObservation.mid
        clientDirection = exchangeObservation.clientDirection
        winningPrice = exchangeObservation.winningPrice
        wonTrade = exchangeObservation.tradeWon
        done = exchangeObservation.episodeFinished

        reward = self._getInventoryPnl(mid)

        inventoryPnl = reward

        #Penalising money made by holding inventory
        reward -= ALPHA_PENALTY * np.min(inventoryPnl, 0)

        if wonTrade:
            self._updateInventory(clientDirection)
            reward += self._getNewBusinessPnl(mid, winningPrice, clientDirection)

        self._updatePreviousMid(mid)

        observation = Observation(self._inventory)

        return observation, reward, done, {}

    def reset(self):
        self._inventory = 0.0
        self._previousMid = self._exchange.getInitialPrice()
        initialObservation = Observation(self._inventory)

        return initialObservation

    def render(self):
        pass

    def _getInventoryPnl(self, mid):
        priceMove = mid - self._previousMid
        return self._inventory * priceMove

    def _getNewBusinessPnl(self, mid, winningPrice, clientDirection):
        if clientDirection == 'clientBuys':
            return INVENTORY_STEP_SIZE * (winningPrice - mid)
        elif clientDirection == 'clientSells':
            return INVENTORY_STEP_SIZE * (mid - winningPrice)
        else:
            raise KeyError("Invalid clientDirection")

    def _updateInventory(self, clientDirection):
        if clientDirection == 'clientBuys':
            self._inventory -= INVENTORY_STEP_SIZE
        elif clientDirection == 'clientSells':
            self._inventory += INVENTORY_STEP_SIZE
        else:
            raise KeyError("invalid clientDirection")

    def _updatePreviousMid(self, newMid):
        self._previousMid = newMid