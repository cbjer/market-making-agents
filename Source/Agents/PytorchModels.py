
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):
    """
    Neural network representation used for the 2 value functions
    """
    def __init__(self, inputSpaceSize, outputSpaceSize, device):
        super(DeepQNetwork, self).__init__()

        self.inputSpaceSize = inputSpaceSize
        self.outputSpaceSize = outputSpaceSize

        # self.createNetwork()
        self.createSequentialNetwork()

    def createNetwork(self):
        self.inputLayer = nn.Linear(self.inputSpaceSize, 128) # Can add weight initialisation later
        self.fullConnected1 = nn.Linear(128, 128)
        self.outputLayer = nn.Linear(128, self.outputSpaceSize)

    def createSequentialNetwork(self):
        hiddenNodes = 128

        self.networkArchitecture = nn.Sequential(
                nn.Linear(self.inputSpaceSize, hiddenNodes),
                nn.ReLU(),
                nn.Linear(hiddenNodes, hiddenNodes),
                nn.ReLU(),
                nn.Linear(hiddenNodes, self.outputSpaceSize)
                )

    def forward(self, x):
        output = self.networkArchitecture(x)
        return output

class ActorNetwork(nn.Module):
    """
    Neural network representation for actor network
    Note, final output has a tanh function -> output in [-1, 1]
    """
    def __init__(self, inputSpaceSize, outputSpaceSize):
        super(ActorNetwork, self).__init__()

        self.inputSpaceSize = inputSpaceSize
        self.outputSpaceSize = outputSpaceSize

        self.createNetwork()

    def createNetwork(self):
        self.inputLayer = nn.Linear(self.inputSpaceSize, 64) # Can add weight initialisation later
        self.fullConnected1 = nn.Linear(64, 128)
        self.outputLayer = nn.Linear(128, self.outputSpaceSize)

    def forward(self, x):
        """
        Maps a state to an action
        """
        x = F.relu(self.inputLayer(x))
        x = F.relu(self.fullConnected1(x))
        x = torch.tanh(self.outputLayer(x))
        return x

class CriticNetwork(nn.Module):
    """
    Neural network representation of the critic network
    """
    def __init__(self, stateDimension, actionDimension):
        super(CriticNetwork, self).__init__()

        self.stateDimension = stateDimension
        self.actionDimension = actionDimension

        self.createNetwork()

    def createNetwork(self):
        self.stateFullyConnected = nn.Linear(self.stateDimension, 64) # Can add weight initialisation later
        self.fc1 = nn.Linear(self.actionDimension + 64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        """
        Maps a state-action pair to a q-value
        """

        #state, action = state.squeeze(), action.squeeze()

        x = F.relu(self.stateFullyConnected(state))
        x = torch.cat((x, action), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class TwinCriticNetwork(nn.Module):
    """
    Twin network for critic
    """
    def __init__(self, stateDimension, actionDimension):
        super(TwinCriticNetwork, self).__init__()
        self.stateDimension = stateDimension
        self.actionDimension = actionDimension

        self.createNetwork()

    def createNetwork(self):
        self.criticOne = CriticNetwork(self.stateDimension, self.actionDimension)
        self.criticTwo = CriticNetwork(self.stateDimension, self.actionDimension)

    def forward(self, state, action):
        x1 = self.criticOne(state, action)
        x2 = self.criticTwo(state, action)
        return x1, x2

    def Q1(self, state, action):
        x = self.criticOne(state, action)
        return x
