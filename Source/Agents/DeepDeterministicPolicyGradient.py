"""
Deep Deterministic Policy Gradient (DDPG) - Used for continuous control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from Source.Agents.Utils.ExperienceReplay import ExperienceReplay

# Parameters
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-4
EXPERIENCE_REPLAY_SIZE = 5000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3

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

class DeepDeterministicPolicyGradient:
    """
    DDPG
    """
    def __init__(self, stateDimensionSize, actionDimensionSize, device):
        self._stateDimensionSize =  stateDimensionSize
        self._actionDimensionSize = actionDimensionSize

        self._actorEvaluationNet = ActorNetwork(stateDimensionSize, actionDimensionSize)
        self._actorTargetNet = ActorNetwork(stateDimensionSize, actionDimensionSize)
        self._actorOptimizer = torch.optim.Adam(self._actorEvaluationNet.parameters(), lr=ACTOR_LEARNING_RATE)

        self._criticEvaluationNet = CriticNetwork(stateDimensionSize, actionDimensionSize)
        self._criticTargetNet = CriticNetwork(stateDimensionSize, actionDimensionSize)
        self._criticOptimizer = torch.optim.Adam(self._criticEvaluationNet.parameters(), lr=CRITIC_LEARNING_RATE)

        self._experienceReplay = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)
        self._device = device

    def storeExperience(self, state, action, reward, done, nextState):
        self._experienceReplay.addExperience(state, action, reward, done, nextState)

    @torch.no_grad()
    def sampleAction(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state).to(self._device), 0)
        self._actorEvaluationNet.eval()
        action = self._actorEvaluationNet(state).cpu().data.numpy()
        self._actorEvaluationNet.train()

        noise = self._getNoise()
        action += noise

        action = np.clip(action, -1.0, 1.0).squeeze()

        assert action.shape == ()
        action = action.item()

        return action

    def learn(self, batchSize=BATCH_SIZE):
        if not self._experienceReplay.isReadyForSampling(batchSize):
            return

        states, actions, rewards, dones, nextStates = self._sampleExperience(batchSize, self._device)

        # Update critic
        nextActions = self._actorTargetNet(nextStates)
        qTargetNexts = self._criticTargetNet(nextStates, nextActions)

        assert qTargetNexts.shape == rewards.shape
        assert dones.shape == rewards.shape

        qTargets = rewards + (GAMMA * qTargetNexts * (1 - dones)).detach()
        qExpected = self._criticEvaluationNet(states, actions)
        criticLoss = F.smooth_l1_loss(qExpected, qTargets)
        self._criticOptimizer.zero_grad()
        criticLoss.backward()
        self._criticOptimizer.step()

        # Update actor
        actionsPredicted = self._actorEvaluationNet(states)
        actorLoss = -1.0 * self._criticEvaluationNet(states, actionsPredicted).mean()
        self._actorOptimizer.zero_grad()
        actorLoss.backward()
        self._actorOptimizer.step()

        # Update target networks
        self._softUpdateParameters(self._criticEvaluationNet, self._criticTargetNet, TAU)
        self._softUpdateParameters(self._actorEvaluationNet, self._actorTargetNet, TAU)

    def _softUpdateParameters(self, localModel, targetModel, tau):
        """
        Performs a soft update of the target parameters
        """
        for targetParameter, localParameter in zip(targetModel.parameters(), localModel.parameters()):
            targetParameter.data.copy_(tau * localParameter.data + ( 1.0 - tau ) * targetParameter.data )

        return

    def _sampleExperience(self, batchSize, device):
        states, actions, rewards, dones, nextStates = self._experienceReplay.sample(batchSize)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).unsqueeze(-1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
        dones = torch.LongTensor(dones).unsqueeze(-1).to(device)
        nextStates = torch.FloatTensor(nextStates).to(device)

        assert dones.shape == rewards.shape
        assert actions.shape == rewards.shape #not necess if action space > 1 - remove when more complicated space

        return states, actions, rewards, dones, nextStates

    def _getNoise(self):
        return 0.0

class DDPGMarketMaker:
    """
    Overlay for market making
    """
    def __init__(self, actionSpaceDimension, stateSpaceDimension, name):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._ddpgNetwork = DeepDeterministicPolicyGradient(stateSpaceDimension, actionSpaceDimension, device)
        self._name = name

    def __str__(self):
        return self._name

    def getSkewAction(self, state):
        state = list(state)
        action = self._ddpgNetwork.sampleAction(state)
        return action

    def inputPostTrade(self, state, action, reward, done, nextState):
        self._ddpgNetwork.storeExperience(state, action, reward, done, nextState)
        self._ddpgNetwork.learn()
