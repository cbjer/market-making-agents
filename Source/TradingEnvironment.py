import numpy as np

from Source.DealerEnvironment import  DealerEnvironment
from torch.utils.tensorboard import SummaryWriter

class TradingEnvironment:
    """
    Final environment for testing
    """
    def __init__(self, listAgents, stockExchange):
        self._numberAgents = len(listAgents)
        self._listAgents = listAgents
        self._stockExchange = stockExchange

        self._prepareDealerEnvironments()
        self._prepareWriters()

    def _prepareDealerEnvironments(self):
        self._dealerEnvironments = []
        for i in range(self._numberAgents):
            name = "dealer_" + str(i)
            dealerEnvironment = DealerEnvironment(dealerId=name, exchange=self._stockExchange)
            self._dealerEnvironments.append(dealerEnvironment)

    def _prepareWriters(self):
        self._listWriters = []
        self._listNames = []
        for agent in self._listAgents:
            name = str(agent)
            writer = SummaryWriter(flush_secs=15, comment=name)
            self._listWriters.append(writer)
            self._listNames.append(name)

        if len(self._listNames) != len(np.unique(self._listNames)):
            raise NameError("Dealers have non unique names")




    def trade(self, totalEpisodes):
        for episodeNumber in range(totalEpisodes):
            self._stockExchange.reset()

            states = [env.reset() for env in self._dealerEnvironments]

            dealerReturns = [0.0] * self._numberAgents

            dealerSkews = [ [] for i in range(self._numberAgents)]
            dealerInventories = [ [state.inventory] for state in states]

            while self._stockExchange.isEpisodeLive():
                skewActions = []

                for i in range(self._numberAgents):
                    state = states[i]
                    skewAction = self._listAgents[i].getSkewAction(state)
                    skewActions.append(skewAction)
                    dealerSkews[i].append(skewAction)
                    self._dealerEnvironments[i].submitSkewToExchange(skewAction)

                self._stockExchange.step()

                nextStates = []

                for i in range(self._numberAgents):
                    state = states[i]
                    action = skewActions[i]
                    nextState, reward, done, _ = self._dealerEnvironments[i].step()
                    nextStates.append(nextState)
                    self._listAgents[i].inputPostTrade(state, action, reward, done, nextState)

                    dealerReturns[i] += reward
                    dealerInventories[i].append(nextState.inventory)

                states = nextStates

            # End of episode
            print("------------------- Episode: {} -------------------".format(episodeNumber))
            for i in range(self._numberAgents):
                writer = self._listWriters[i]
                name = self._listNames[i]
                dealerReturn = dealerReturns[i]
                meanInventory = np.mean(dealerInventories[i])
                writer.add_scalar("return", dealerReturn, episodeNumber)
                writer.add_scalar("inventory", meanInventory, episodeNumber)

                print("Dealer: {}, Return: {}".format(name, dealerReturn))

        print("End of trading")
        return







