import torch
import numpy as np
from collections import deque, namedtuple

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'nextState'])

BatchExperience = namedtuple('BatchExperience', field_names=['batchStates', 'batchActions', 'batchRewards', 'batchDones', 'batchNextStates'])

class ExperienceReplay:
    """
    Allows storage and sampling of experience
    Using it for discrete action spaces for now
    """
    def __init__(self, maxSize):
        self._maxSize = maxSize
        self._storage = deque(maxlen=maxSize)

    def __len__(self):
        return len(self._storage)

    def addExperience(self, state, action, reward, done, nextState):
        experience = Experience(state, action, reward, done, nextState)
        self._storage.append(experience)

    def sample(self, batchSize):
        indices = np.random.choice(len(self._storage), batchSize, replace=False)

        states, actions, rewards, dones, nextStates = zip(*[self._storage[idx] for idx in indices])

        batchExperience = BatchExperience(states, actions, rewards, dones, nextStates)

        return batchExperience

    def isReadyForSampling(self, batchSize):
        return len(self._storage) >= batchSize