"""
Deep Deterministic Policy Gradient (DDPG) - Used for continuous control
"""

import torch
import torch.nn.functional as F
import numpy as np

from Source.Agents.Utils.ExperienceReplay import ExperienceReplay
from Source.Agents.PytorchModels import ActorNetwork, CriticNetwork
import Source.Agents.AgentConstructionUtils as AgentUtils

# Parameters
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-4
EXPERIENCE_REPLAY_SIZE = 5000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3

NOISE_START = 1.0
NOISE_DECAY = 0.99999
NOISE_MIN = 0.02

class DeepDeterministicPolicyGradient:
    """
    DDPG
    """
    def __init__(self, stateDimensionSize, actionDimensionSize, device):
        self._stateDimensionSize = stateDimensionSize
        self._actionDimensionSize = actionDimensionSize

        self._actorEvaluationNet = ActorNetwork(stateDimensionSize, actionDimensionSize).to(device)
        self._actorTargetNet = ActorNetwork(stateDimensionSize, actionDimensionSize).to(device)
        self._actorOptimizer = torch.optim.Adam(self._actorEvaluationNet.parameters(), lr=ACTOR_LEARNING_RATE)

        self._criticEvaluationNet = CriticNetwork(stateDimensionSize, actionDimensionSize).to(device)
        self._criticTargetNet = CriticNetwork(stateDimensionSize, actionDimensionSize).to(device)
        self._criticOptimizer = torch.optim.Adam(self._criticEvaluationNet.parameters(), lr=CRITIC_LEARNING_RATE)

        self._experienceReplay = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)
        self._device = device

        self._noiseStd = NOISE_START

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

        states, actions, rewards, dones, nextStates = AgentUtils.sampleExperience(self._experienceReplay, batchSize, self._device)

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
        AgentUtils.softUpdateModelParameters(self._criticEvaluationNet, self._criticTargetNet, TAU)
        AgentUtils.softUpdateModelParameters(self._actorEvaluationNet, self._actorTargetNet, TAU)

    def _getNoise(self):
        noise = np.random.normal(loc=0.0, scale=self._noiseStd)
        self._noiseStd = np.max([self._noiseStd * NOISE_DECAY, NOISE_MIN])
        return noise

class DDPGMarketMaker:
    """
    Overlay for market making
    """
    def __init__(self, actionSpaceDimension, stateSpaceDimension, name):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("DDPG using {}".format(device))

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
