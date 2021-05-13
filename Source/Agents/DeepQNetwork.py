"""
Requires a discrete actions space
"""
from Source.Agents.Utils.ExperienceReplay import ExperienceReplay

import torch
import numpy as np

LEARNING_RATE = 0.0005
WEIGHT_TRANSFER_INTERVALS = 5000
EXPERIENCE_REPLAY_SIZE = 5000
EPSILON_START = 0.5
EPSILON_ALPHA = 0.9999
EPSILON_MINIMUM = 0.01
BATCH_SIZE = 128
GAMMA = 0.95

class Net(torch.nn.Module):
    """
    Neural network representation used for the 2 value functions
    """
    def __init__(self, inputSpaceSize, outputSpaceSize, device):
        super(Net, self).__init__()

        self.inputSpaceSize = inputSpaceSize
        self.outputSpaceSize = outputSpaceSize
        self.device = device

        # self.createNetwork()
        self.createSequentialNetwork()

    def createNetwork(self):
        self.inputLayer = torch.nn.Linear(self.inputSpaceSize, 128) # Can add weight initialisation later
        self.fullConnected1 = torch.nn.Linear(128, 128)
        self.outputLayer = torch.nn.Linear(128, self.outputSpaceSize)

    def createSequentialNetwork(self):
        hiddenNodes = 128
        self.networkArchitecture = torch.nn.Sequential(
                torch.nn.Linear(self.inputSpaceSize, hiddenNodes),
                torch.nn.ReLU(),
                torch.nn.Linear(hiddenNodes, hiddenNodes),
                torch.nn.ReLU(),
                torch.nn.Linear(hiddenNodes, self.outputSpaceSize)
                )
        if self.device == 'cuda':
            print("Activating cuda for Net")
            self.networkArchitecture.cuda()

    def forward(self, x):
        """
        x = self.inputLayer(x)
        x = torch.nn.functional.relu(x)
        x = self.fullConnected1(x)
        x = torch.nn.functional.relu(x)
        output = self.outputLayer(x)
        """
        output = self.networkArchitecture(x)
        return output

class DeepQNetwork:
    """
    Replica of Deepmind's DQN structure
    Requires a discrete action space
    Inspired by Sweetice implementation
    """
    def __init__(self, actionSpaceSize, stateSpaceSize, device):
        self._actionSpaceSize = actionSpaceSize
        self._stateSpaceSize = stateSpaceSize
        self._epsilon = EPSILON_START
        self._experienceReplay = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)

        self._evaluationNet = Net(self._stateSpaceSize, self._actionSpaceSize, device)
        self._targetNet = Net(self._stateSpaceSize, self._actionSpaceSize, device)

        self._learnCounter = 0
        self._weightsTransferIntervals = WEIGHT_TRANSFER_INTERVALS
        self._optimizer = torch.optim.Adam(self._evaluationNet.parameters(), lr=LEARNING_RATE)
        self._lossFunction = torch.nn.MSELoss()
        self._device = device

    def sampleEpsilonGreedyActionIndex(self, state):
        if np.random.random() < self._epsilon:
            return self._sampleRandomActionIndex()
        else:
            return self.sampleGreedyActionIndex(state)

    @torch.no_grad()
    def sampleGreedyActionIndex(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state).to(self._device), 0)
        actionValues = self._evaluationNet.forward(state)
        maxAction = torch.max(actionValues, 1)
        maxActionIndex = maxAction[1].cpu().data.numpy()
        maxActionIndex = maxActionIndex[0]
        return maxActionIndex

    def _sampleRandomActionIndex(self):
        randomIndex = np.random.randint(0, self._actionSpaceSize)
        return randomIndex

    def storeExperience(self, state, actionIndex, reward, done, nextState):
        self._experienceReplay.addExperience(state, actionIndex, reward, done, nextState)

    def learn(self, batchSize=BATCH_SIZE):
        if batchSize > len(self._experienceReplay):
            return

        if self._learnCounter % self._weightsTransferIntervals == 0:
            self._learnCounter = 1
            self._copyEvalautionWeightsToTarget()

        self._learnCounter += 1
        self._epsilon = np.max([EPSILON_ALPHA * self._epsilon, EPSILON_MINIMUM])

        states, actionIndexes, rewards, dones, nextStates = self._sampleExperience(batchSize)

        qActionValues = self._evaluationNet(states).gather(1, actionIndexes)
        nextStateQValues = self._targetNet(nextStates).detach()
        nextStateMaxActionValue = nextStateQValues.max(1)[0].view(batchSize, 1)

        assert nextStateMaxActionValue.shape == rewards.shape
        assert dones.shape == rewards.shape

        qTarget = rewards + GAMMA * nextStateMaxActionValue * ( 1  - dones )

        assert qTarget.shape == qActionValues.shape

        loss = self._lossFunction(qActionValues, qTarget)
        self._optimizer.zero_grad()
        loss.backward()

        for param in self._evaluationNet.parameters():
            param.grad.data.clamp_(-1, 1)

        self._optimizer.step()

    def _copyEvalautionWeightsToTarget(self):
        self._targetNet.load_state_dict(self._evaluationNet.state_dict(), strict=True)

    def _sampleExperience(self, batchSize):
        states, actionIndexes, rewards, dones, nextStates = self._experienceReplay.sample(batchSize, self._device)
        return states, actionIndexes, rewards, dones, nextStates

    def getEpsilon(self):
        return self._epsilon

class DQNMarketMaker:
    """
    Specific for the market making environment
    """
    def __init__(self, actionSpace, stateSpaceSize):
        self.actionSpace = actionSpace
        self.actionSpaceSize = len(actionSpace)
        self.stateSpaceSize = stateSpaceSize

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("DQN using {}".format(device))

        self._deepQNetwork = DeepQNetwork(self.actionSpaceSize, self.stateSpaceSize, device=device)

    def getSkewAction(self, state):
        state = list(state)
        actionIndex = self._deepQNetwork.sampleEpsilonGreedyActionIndex(state)
        return self.actionSpace[actionIndex], actionIndex

    def inputPostTrade(self, state, actionIndex, reward, done, nextState):
        self._deepQNetwork.storeExperience(state, actionIndex, reward, done, nextState)
        self._deepQNetwork.learn()

    def getEpsilon(self):
        return self._deepQNetwork.getEpsilon()