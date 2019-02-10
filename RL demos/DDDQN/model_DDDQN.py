import random
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import Transition


class DQN(nn.Module):
    """ DDDQN model
        Dueling DQN Implementation
        2 FC Layers to calculate Q(s, a):
        Value stream calculates V(s)
        Advantage stream calculates A(s, a) for each action a
        Aggregation layer: Q(s, a) = V(s) + (A(s, a) - mean value across actions(A(s, )))
    """

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        # 3 FC layers to encode observations
        self.FC1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.FC2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.FC3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)

        # Dueling DQN: two streams
        # Value stream: output dimension - 1
        self.VFC1 = nn.Linear(64, 32)
        self.vbn1 = nn.BatchNorm1d(32)
        self.VFC2 = nn.Linear(32, 16)
        self.vbn2 = nn.BatchNorm1d(16)
        self.VFC3 = nn.Linear(16, 1)
        # Action stream: output dimension - 2
        self.AFC1 = nn.Linear(64, 32)
        self.abn1 = nn.BatchNorm1d(32)
        self.AFC2 = nn.Linear(32, 16)
        self.abn2 = nn.BatchNorm1d(16)
        self.AFC3 = nn.Linear(16, output_size)

        self.elu = nn.ELU()

    def forward(self, x):
        x = self.bn1(self.elu(self.FC1(x)))
        x = self.bn2(self.elu(self.FC2(x)))
        x = self.bn3(self.elu(self.FC3(x)))
        # print("x shape: ", x.shape)

        # Value stream:
        v = self.vbn1(self.elu(self.VFC1(x)))
        v = self.vbn2(self.elu(self.VFC2(v)))
        v = self.elu(self.VFC3(v))
        # print("v shape:", v.shape)

        # Action sream:
        a = self.abn1(self.elu(self.AFC1(x)))
        a = self.abn2(self.elu(self.AFC2(a)))
        a = self.elu(self.AFC3(a))
        # print("a shape:", a.shape)

        # Aggregate:
        # print("a mean shape: ", a.mean(dim=-1).shape)
        # print("a substracted shape:", (a - a.mean(dim=-1, keepdim=True)).shape)
        q = v + (a - a.mean(dim=-1, keepdim=True))
        # print("q shape:", q.shape)

        return q


steps_done = 0


def select_action(state, policy_net, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, device='cuda'):
    """Epsilon-greedy selection of the optimal action given the state"""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)  # Annealing
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest value for column of each row.
            # second column on max result is index of where max element was found,
            # so we pick action with the larger expected reward

            # DOUBLE DQN implementation:
            # . we use the online policy net to greedily select the action
            # . and the target net to estimate the Q-value
            # . so NO CHANGE HERE
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


def select_action_for_evaluation(state, policy_net):
    """Deterministic selection of the optimal action given the state.
       Used in evaluation
    """
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)


def optimize_model(batch_size, memory, policy_net, target_net, optimizer, GAMMA=0.999, device='cuda'):
    """Optimize the model for one step
       Return mini-batch loss
    """

    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch. This converts batch-array of Transitions to Transition of batch-arrays
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
    # These are the actions which would've been taken for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    # Expected values of actions for non_final_next_states are computed
    # This is merged based on the mask, such that we'll have either the expected state value or 0
    # in case the state was final

    # DOUBLE DQN implementation:
    # . we use the online policy net to greedily select the action
    # . and the target net to estimate the Q-value
    next_state_values = torch.zeros(batch_size, device=device)
    next_action_policynet_decisions = policy_net(non_final_next_states).max(1)[1]
    non_final_next_state_targetnet_values = target_net(non_final_next_states) \
                                                .gather(1, next_action_policynet_decisions.view(-1, 1).repeat(1, 2))[:,
                                            0]
    next_state_values[non_final_mask] = non_final_next_state_targetnet_values.detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # Return minibatch huber loss
    return loss.item()


# Utils for saving and loading checkpoints

def save_checkpoint(file_dir, policy_net, target_net, optimizer, memory, i_episode, episode_rewards, episode_loss):
    save_dict = {"policy_net": policy_net.state_dict(),
                 "target_net": target_net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "memory": memory,
                 "i_episode": i_episode,
                 "episode_rewards": episode_rewards,
                 "episode_loss": episode_loss}
    # Create the directory if not exist
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    # Delete the file if the file already exist
    file_name = os.path.join(file_dir, "ckpt_eps%d.pt" % i_episode)
    try:
        os.remove(file_name)
    except OSError:
        pass
    # Save the file
    torch.save(save_dict, file_name)


def load_checkpoint(file_dir, i_episode, input_size, output_size, device='cuda'):
    checkpoint = torch.load(os.path.join(file_dir, "ckpt_eps%d.pt" % i_episode))

    policy_net = DQN(input_size, output_size).to(device)
    policy_net.load_state_dict(checkpoint["policy_net"])
    policy_net.train()

    target_net = DQN(input_size, output_size).to(device)
    target_net.load_state_dict(checkpoint["target_net"])
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    optimizer.load_state_dict(checkpoint["optimizer"])

    memory = checkpoint["memory"]

    i_episode = checkpoint["i_episode"]
    episode_rewards = checkpoint["episode_rewards"]
    episode_loss = checkpoint["episode_loss"]

    return policy_net, target_net, optimizer, memory, i_episode, episode_rewards, episode_loss
