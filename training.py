import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import gym
import numpy as np
import torch
import trading_gym
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import count
import os
from model.model import Encoder, PolicyNet, optimize_model
from model.save_and_load import save_checkpoint, load_checkpoint
from data.dataset import StockDataset
from model.utils import plot_durations

#######################  Parameters  ##############################

# Dataset parameters
csv_file = "data/AAPL-Updated.csv"

window_len = 30  # Number of trading days in a window
dataset_size = 10

earliest_date = "01-02-2014"
latest_date = "01-18-2019"
datestr_format = "%m-%d-%Y"

# Model hyperparameters
encoder_input_size = 3
state_size = 4
num_actions = 3

# ckpt_dir = "simplePG_Adam_%s_obs_checkpoints/" % (env_name)
save_ckpt_interval = 10

# Environment parameter
env_name = 'SeriesEnv-v0'

# Training parameters
# num_episodes = 1000
i_epoch = 460      # This would determine which checkpoint to load, if the checkpoint exists
batch_size = 32
learning_rate = 0.0003
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# # Rendering and recording options
# render = False
plot = False
#
# render_each_episode = False     # Whether to render each episode
#                                 #   If set to true, then each episode the agent ever endure will be rendered
#                                 #   Otherwise, only each episode at the start of each epoch will be rendered
#                                 #   Note: each epoch has exactly 1 model update and batch_size episodes
#
# # record_each_episode_stats = False   # Whether to record the statistics of each episode
#                                     #   If set to true, then each episode the agent ever endure will be recorded
#                                     #   Otherwise, only each episode at the start of each epoch will be recorded
#
num_avg_epoch = 5       # The number of epochs to take for calculating average stats

###################################################################


# Turn on pyplot's interactive mode
# VERY IMPORTANT because otherwise training stats plot will hault
plt.ion()

# Create OpenAI gym environment
env = gym.make(env_name)

# Initialize dataset and dataloader
dataset = StockDataset(csv_file, earliest_date, latest_date, datestr_format, window_len, dataset_size)
dataloader = DataLoader(dataset)

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current usable device is: ", device)

# Create the model
encoder = Encoder(encoder_input_size)
policy_net = PolicyNet(state_size, num_actions)

# Set up optimizer - Minimal
optimizer = optim.Adam(policy_net.parameters())

###################################################################
# Start training

# Dictionary for extra training information to save to checkpoints
training_info = {"epoch mean durations" : [],
                 "epoch mean rewards" : [],
                 "max reward achieved": 0,
                 "past %d epochs mean reward" %  (num_avg_epoch): 0,}

# Batch that records trajectories
batch_log_prob = []
batch_rewards = []

# TODO: Initialize trading gym

while True:

    epoch_durations = []
    epoch_rewards = []


    # for i_epoch, sample_batched in enumerate(dataloader):

    # Get a random batch
    idx = np.random.randint(0, dataset_size)

    # Use the first half to pass through encoder: Open, Volume, Percent-Change
    half_len = dataset[idx]['Open'].shape[0] // 2
    open = dataset[idx]['Open'][:half_len]
    volume = dataset[idx]['Volume'][:half_len]
    pc = dataset[idx]['Percent-Change'][:half_len]

    # Stack to form tensor input
    open = torch.tensor(open, device=device, dtype=torch.float32)
    volume = torch.tensor(volume, device=device, dtype=torch.float32)
    pc = torch.tensor(pc, device=device, dtype=torch.float32)

    batch = torch.stack([open, volume, pc], dim=1)
    batch = batch.unsqueeze(dim=0)      # spare the batch dimensions

    # Propagate through encoder
    encoding = encoder(batch)

    pass

    # Pass through encoder

    for i_episode in range(batch_size):
        # Every save_ckpt_interval, Check if there is any checkpoint.
        # If there is, load checkpoint and continue training
        # Need to specify the i_episode of the checkpoint intended to load
        # if i_epoch % save_ckpt_interval == 0 and os.path.isfile(os.path.join(ckpt_dir, "ckpt_eps%d.pt" % i_epoch)):
        #     policy_net, optimizer, training_info = load_checkpoint(ckpt_dir, i_epoch, layer_sizes, action_lim, device=device)

        # Initialize the environment and state
        observation = env.reset()
        current_state = torch.tensor([observation], device=device, dtype=torch.float32)

        traj_log_prob = None
        traj_reward = None

        running_reward = 0
        episode_durations = []
        episode_rewards = []

        for t in count():
            # Make sure that policy net is in training mode
            policy_net.train()

            # Sample an action given the current state
            action, log_prob = policy_net(current_state)

            # Interact with the environment
            observation, reward, done, _ = env.step(action.to('cpu').numpy())

            # Record action log_prob
            if traj_log_prob is None:
                traj_log_prob = log_prob
            else:
                traj_log_prob = torch.cat([traj_log_prob, log_prob])

            # Record reward
            running_reward += reward
            reward = torch.tensor([reward], device=device, dtype=torch.float)
            if traj_reward is None:
                traj_reward = reward
            else:
                traj_reward = torch.cat([traj_reward, reward])

            # Update state
            if not done:
                next_state = torch.tensor([observation], device=device, dtype=torch.float32)
            else:
                next_state = done

            current_state = next_state

            if done:
                # Load and print episode stats after each episode ends
                episode_durations.append(t + 1)
                episode_rewards.append(running_reward)
                if running_reward > training_info["max reward achieved"]:
                    training_info["max reward achieved"] = running_reward

                print("=============  Epoch: %d, Episode: %d  =============" % (i_epoch + 1, i_episode + 1))
                print("Episode reward: %f" % episode_rewards[-1])
                print("Episode durations: %d" % episode_durations[-1])
                print("Episode duration: %d" % (t + 1))
                print("Max reward achieved: %f" %  training_info["max reward achieved"])

                # Check if the problem is solved
                #  CartPole standard: average reward for the past 100 episode above 195
                # if training_info["past 100 episodes mean reward"] > 195:
                #     print("\n\n\t Problem Solved !!!\n\n\n")


                break

        # Store trajectory
        batch_log_prob.append(traj_log_prob)
        batch_rewards.append(traj_reward)

        epoch_durations.append(sum(episode_durations))
        epoch_rewards.append(sum(episode_rewards))

    # At the end of each epoch
    # Optimize the model for one step after collecting enough trajectories
    # And record epoch stats
    optimize_model(policy_net, batch_log_prob, batch_rewards, optimizer, GAMMA, device=device)

    # Clear trajectories batch
    batch_log_prob = []
    batch_rewards = []

    # Record stats
    training_info["epoch mean durations"].append(sum(epoch_durations) / batch_size)
    training_info["epoch mean rewards"].append(sum(epoch_rewards) / batch_size)
    if (i_epoch + 1) % num_avg_epoch:
        training_info["past %d epochs mean reward" %  (num_avg_epoch)] = \
            (sum(training_info["epoch mean rewards"][-num_avg_epoch:]) / num_avg_epoch) \
                if len(training_info["epoch mean rewards"]) >= num_avg_epoch else 0


    # Plot stats
    if plot:
        plot_durations(training_info["epoch mean rewards"])

    # Update counter
    i_epoch += 1

    # Every save_ckpt_interval, save a checkpoint according to current i_episode.
    # if (i_epoch) % save_ckpt_interval == 0:
    #     save_checkpoint(ckpt_dir, policy_net, optimizer, i_epoch, learning_rate=learning_rate,
    #                     **training_info)
    #
