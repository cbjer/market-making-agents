"""
Useful utils for constructing agents
"""
import torch

def softUpdateModelParameters(localModel, targetModel, tau):
    """
    Copies the weights from the local model to the target model as a soft-update for small tau
    :param localModel: Pytorch model
    :param targetModel: Pytorch model
    :param tau: float (small)
    :return: None
    """
    for targetParameter, localParameter in zip(targetModel.parameters(), localModel.parameters()):
        targetParameter.data.copy_(tau * localParameter.data + (1.0 - tau) * targetParameter.data)

    return

def sampleExperience(experienceReplay, batchSize, device):
    """
    returns sampled experience from a replay buffer in pytorch tensor format
    :param experienceReplay: ExperienceReplay object
    :param batchSize: int
    :param device: "cuda" or "cpu"
    :return: states, actions, rewards, dones, nextStates
    """
    states, actions, rewards, dones, nextStates = experienceReplay.sample(batchSize)

    states = torch.FloatTensor(states).to(device)
    actions = torch.FloatTensor(actions).unsqueeze(-1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(-1).to(device)
    nextStates = torch.FloatTensor(nextStates).to(device)

    assert dones.shape == rewards.shape
    assert actions.shape == rewards.shape  # not necess if action space > 1 - remove when more complicated space

    return states, actions, rewards, dones, nextStates






