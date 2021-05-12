import torch
import numpy as np
from collections import deque, namedtuple

class ExperienceReplay:
    """
    Allows storage and sampling of experience
    Using it for discrete action spaces for now
    """
    def __init__(self, maxSize):
        self._maxSize = maxSize
        self._storage = deque(maxlen=maxSize)
        self._experienceFactory = namedtuple('Experience', field_names=['state', 'actionIndex', 'reward', 'done', 'nextState'])

    def __len__(self):
        return len(self._storage)

    def addExperience(self, state, actionIndex, reward, done, nextState):
        experience = self._experienceFactory(state, actionIndex, reward, done, nextState)
        self._storage.append(experience)

    def sample(self, batchSize, device='cpu'):
        indices = np.random.choice(len(self._storage), batchSize, replace=False)

        states, actionIndexes, rewards, dones, nextStates = zip(*[self._storage[idx] for idx in indices])

        # Can we be more compute efficient here?
        states = torch.FloatTensor(states).to(device)
        actionIndexes = torch.LongTensor(actionIndexes).unsqueeze(-1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
        dones = torch.LongTensor(dones).unsqueeze(-1).to(device)
        nextStates = torch.FloatTensor(nextStates).to(device)

        assert dones.shape == rewards.shape
        assert actionIndexes.shape == rewards.shape

        return states, actionIndexes, rewards, dones, nextStates