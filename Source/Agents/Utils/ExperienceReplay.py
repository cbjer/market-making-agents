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

    def sample(self, batchSize):
        indices = np.random.choice(len(self._storage), batchSize, replace=False)

        states, actionIndexes, rewards, dones, nextStates = zip(*[self._storage[idx] for idx in indices])

        # Can we be more compute efficient here?
        states = torch.FloatTensor(states)
        actionIndexes = torch.LongTensor(actionIndexes).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.LongTensor(dones)
        nextStates = torch.FloatTensor(nextStates)

        return states, actionIndexes, rewards, dones, nextStates