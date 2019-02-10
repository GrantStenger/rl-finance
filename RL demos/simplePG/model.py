import random
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    """
        Simple Multilayer Perceptron for Policy Gradient
    """

    def __init__(self, layer_sizes):
        super(PolicyNet, self).__init__()

        # self.FC1 = nn.Linear(input_size, 32)
        # self.FC2 = nn.Linear(32, 32)
        # self.FC3 = nn.Linear(32, output_size)

        # Store the network layers in a ModuleList
        self.layers = nn.ModuleList()
        input_size = layer_sizes[0]
        for output_size in layer_sizes[1:]:
            self.layers.append(nn.Linear(input_size, output_size))
            input_size = output_size

        self.Elu = nn.ELU()


    def forward(self, x):
        # Forward propagation
        for layer in self.layers[:-1]:
            x = self.Elu(layer(x))

        # Compute the logits that will be later used to compute action probability
        logits = self.layers[-1](x)

        # If the model is in evaluation mode, deterministically select the best action
        # Else, return an action sampled and the log of its probability
        if not self.training:
            return logits.argmax()
        else:
            # Instantiate a Categorical (multinomial) distribution that can be used to sample action
            #   or compute the action log-probabilities
            m = Categorical(logits=logits)

            action = m.sample()
            log_prob = m.log_prob(action)
            return action, log_prob


def optimize_model(policy_net, batch_log_prob, batch_rewards, optimizer, GAMMA=0.999, device='cuda'):
    """ Optimize the model for one step"""

    # Obtain batch size
    batch_size = len(batch_log_prob)

    # Calculate weight
    # Simple Policy Gradient: Use trajectory Reward To Go
    batch_weight = []
    for rewards in batch_rewards:
        n = rewards.shape[0]
        rtg = torch.zeros(n, device=device)
        for i in reversed(range(n)):
            rtg[i] = rewards[i] + (GAMMA * rtg[i+1] if i + 1 < n else 0)
        batch_weight.append(rtg)

    # Calculate grad-prob-log
    loss = None
    for i in range(batch_size):
        if loss is None:
            loss =  - torch.sum(batch_log_prob[i] * batch_weight[i])
        else:
            loss += - torch.sum(batch_log_prob[i] * batch_weight[i])

    loss = loss / torch.tensor(batch_size, device=device)
    # Gradient Ascent
    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
