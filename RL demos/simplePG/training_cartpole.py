import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import os
from model import PolicyNet, optimize_model
from save_and_load import save_checkpoint, load_checkpoint
from utils import plot_durations


#######################  Parameters  ##############################

# Environment parameter
env_name = "MountainCar-v0"
is_unwrapped = False

# Model hyperparameters
input_size = 2      # Size of state
output_size = 3     # Number of discrete actions
layer_sizes = [input_size, 32, 32, output_size]        # The MLP network architecture
ckpt_dir = "simplePG_Adam_%s_obs_checkpoints/" % (env_name)
save_ckpt_interval = 10

# Training parameters
# num_episodes = 1000
i_epoch = 0      # This would determine which checkpoint to load, if the checkpoint exists
batch_size = 32
learning_rate = 0.0003
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# Rendering and recording options
render_each_episode = False     # Whether to render each episode
                                #   If set to true, then each episode the agent ever endure will be rendered
                                #   Otherwise, only each episode at the start of each epoch will be rendered
                                #   Note: each epoch has exactly 1 model update and batch_size episodes

# record_each_episode_stats = False   # Whether to record the statistics of each episode
                                    #   If set to true, then each episode the agent ever endure will be recorded
                                    #   Otherwise, only each episode at the start of each epoch will be recorded

num_avg_epoch = 5       # The number of epochs to take for calculating average stats

###################################################################


# Turn on pyplot's interactive mode
# VERY IMPORTANT because otherwise training stats plot will hault
plt.ion()

# Create OpenAI gym environment
env = gym.make(env_name)
if is_unwrapped:
    env = env.unwrapped

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current usable device is: ", device)

# Create the model
policy_net = PolicyNet(layer_sizes).to(device)

# Set up optimizer - Minimal
optimizer = optim.Adam(policy_net.parameters())
# optimizer = optim.SGD(policy_net.parameters(), lr=learning_rate)

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

while True:

    finished_rendering_this_epoch = False
    epoch_durations = []
    epoch_rewards = []
    for i_episode in range(batch_size):
        # Every save_ckpt_interval, Check if there is any checkpoint.
        # If there is, load checkpoint and continue training
        # Need to specify the i_episode of the checkpoint intended to load
        if i_epoch % save_ckpt_interval == 0 and os.path.isfile(os.path.join(ckpt_dir, "ckpt_eps%d.pt" % i_epoch)):
            policy_net, optimizer, training_info = load_checkpoint(ckpt_dir, i_epoch, layer_sizes, device=device)

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
            observation, reward, done, _ = env.step(action.item())

            # Render this episode
            if render_each_episode or (not finished_rendering_this_epoch):
                env.render()

            # Record action log_prob
            if traj_log_prob is None:
                traj_log_prob = log_prob
            else:
                traj_log_prob = torch.cat([traj_log_prob, log_prob])

            # Record reward
            running_reward += reward
            reward = torch.tensor([reward], device=device)
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

                # Decide whether to render next episode
                if not(render_each_episode):
                    finished_rendering_this_epoch = True

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

    # Reset Flags
    if not(render_each_episode):
        finished_rendering_this_epoch = False

    # Record stats
    training_info["epoch mean durations"].append(sum(epoch_durations) / batch_size)
    training_info["epoch mean rewards"].append(sum(epoch_rewards) / batch_size)
    if (i_epoch + 1) % num_avg_epoch:
        training_info["past %d epochs mean reward" %  (num_avg_epoch)] = \
            (sum(training_info["epoch mean rewards"][-num_avg_epoch:]) / num_avg_epoch) \
                if len(training_info["epoch mean rewards"]) >= num_avg_epoch else 0


    # Plot stats
    plot_durations(training_info["epoch mean rewards"])

    # Update counter
    i_epoch += 1

    # Every save_ckpt_interval, save a checkpoint according to current i_episode.
    if (i_epoch) % save_ckpt_interval == 0:
        save_checkpoint(ckpt_dir, policy_net, optimizer, i_epoch, learning_rate=learning_rate,
                        **training_info)

