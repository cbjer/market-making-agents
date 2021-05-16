"""
Implementation of the TD3 Algorithm
Twin Delayed Deep Deterministic Policy Gradient
"""

import torch
import torch.nn.functional as F
import numpy as np

from Source.Agents.Utils.ExperienceReplay import ExperienceReplay
from Source.Agents.PytorchModels import ActorNetwork, TwinCriticNetwork
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

POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
MIN_ACTION = -1.0
MAX_ACTION = 1.0
ACTOR_POLICY_UPDATE_FREQUENCY = 2

class TwinDelayedDDPG:
    """
    AKA TD3
    """
    def __init__(self, stateDimensionSize, actionDimensionSize, device):
        self._stateDimensionSize = stateDimensionSize
        self._actionDimensionSize = actionDimensionSize

        self._actorEvaluationNet = ActorNetwork(stateDimensionSize, actionDimensionSize).to(device)
        self._actorTargetNet = ActorNetwork(stateDimensionSize, actionDimensionSize).to(device)
        self._actorOptimizer = torch.optim.Adam(self._actorEvaluationNet.parameters(), lr=ACTOR_LEARNING_RATE)

        self._criticEvaluationNet = TwinCriticNetwork(stateDimensionSize, actionDimensionSize).to(device)
        self._criticTargetNet = TwinCriticNetwork(stateDimensionSize, actionDimensionSize).to(device)
        self._criticOptimizer = torch.optim.Adam(self._criticEvaluationNet.parameters(), lr=CRITIC_LEARNING_RATE)

        self._experienceReplay = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)
        self._device = device

        self._noiseStd = NOISE_START
        self._step = 1

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

    def learn(self, batchSize=BATCH_SIZE, policyNoise=POLICY_NOISE):
        if not self._experienceReplay.isReadyForSampling(batchSize):
            return

        states, actions, rewards, dones, nextStates = AgentUtils.sampleExperience(self._experienceReplay, batchSize, self._device)

        # Apply noise to the actions
        noise = actions.data.normal_(0, policyNoise).to(self._device)
        noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
        nextActions = (self._actorTargetNet(nextStates) + noise).clamp(MIN_ACTION, MAX_ACTION)

        # Update critic
        qTargetNexts1, qTargetNexts2 = self._criticTargetNet(nextStates, nextActions)
        qTargetNexts = torch.min(qTargetNexts1, qTargetNexts2)

        assert qTargetNexts.shape == rewards.shape
        assert dones.shape == rewards.shape

        qTargets = rewards + (GAMMA * qTargetNexts * (1 - dones)).detach()
        qExpected1, qExpected2 = self._criticEvaluationNet(states, actions)

        criticLoss = F.smooth_l1_loss(qExpected1, qTargets) + F.smooth_l1_loss(qExpected2, qTargets)

        self._criticOptimizer.zero_grad()
        criticLoss.backward()
        self._criticOptimizer.step()

        if self._step % ACTOR_POLICY_UPDATE_FREQUENCY == 0:
            self._step = 1
        else:
            self._step += 1
            return

        # Update actor
        actionsPredicted = self._actorEvaluationNet(states)
        actorLoss = -1.0 * self._criticEvaluationNet.Q1(states, actionsPredicted).mean()

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

class TD3MarketMaker:
    """
    Overlay for market making
    """
    def __init__(self, actionSpaceDimension, stateSpaceDimension, name):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("TD3 using {}".format(device))

        self._td3Network = TwinDelayedDDPG(stateSpaceDimension, actionSpaceDimension, device)
        self._name = name

    def __str__(self):
        return self._name

    def getSkewAction(self, state):
        state = list(state)
        action = self._td3Network.sampleAction(state)
        return action

    def inputPostTrade(self, state, action, reward, done, nextState):
        self._td3Network.storeExperience(state, action, reward, done, nextState)
        self._td3Network.learn()
