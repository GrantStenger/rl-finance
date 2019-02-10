import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import gym
import torch
import torch.optim as optim
from itertools import count
import os
from utils import ReplayMemory, plot_durations
from model_DDDQN import DQN, select_action, optimize_model, save_checkpoint, load_checkpoint

# Turn on pyplot's interactive mode
# VERY IMPORTANT because otherwise training stats plot will hault
plt.ion()

# Create CartPole gym environment
env = gym.make('Pendulum-v0').unwrapped

# Get device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print("Current usable device is: ", device)

########################################
# Model hyperparameters
input_size = 3      # Size of state
output_size = 1     # Number of discrete actions
batch_size = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
target_update = 10

# Create the models
policy_net = DQN(input_size, output_size).to(device)
target_net = DQN(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Set up replay memory
memory = ReplayMemory(10000)

# Set up optimizer
optimizer = optim.Adam(policy_net.parameters())

########################################
# Start training
num_episodes = 500
ckpt_dir = "DDDQN_CartPoleV1_obs_checkpoints/"
save_ckpt_interval = 100

episode_rewards = []
episode_loss = []
i_episode = 0

policy_net.train()

while True:
    # Every save_ckpt_interval, Check if there is any checkpoint.
    # If there is, load checkpoint and continue training
    # Need to specify the i_episode of the checkpoint intended to load
    if i_episode % save_ckpt_interval == 0 and os.path.isfile(os.path.join(ckpt_dir, "ckpt_eps%d.pt" % i_episode)):
        policy_net, target_net, optimizer, memory, i_episode, episode_rewards, episode_loss = \
            load_checkpoint(ckpt_dir, i_episode, input_size=input_size, output_size=output_size, device=device)

    # Initialize the environment and state
    observation = env.reset()
    current_state = torch.tensor([observation], device=device, dtype=torch.float32)

    running_loss = 0
    running_reward = 0
    for t in count():
        # Select and perform an action
        # Turn policy_net into evaluation mode to select an action given a single state
        policy_net.eval()

        action = select_action(current_state, policy_net, EPS_START=EPS_START, EPS_END=EPS_END, EPS_DECAY=EPS_DECAY ,
                               device=device)

        observation, reward, done, _ = env.step([action.item()])
        env.render()

        running_reward += reward
        reward = torch.tensor([reward], device=device)

        if not done:
            next_state = torch.tensor([observation], device=device, dtype=torch.float32)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(current_state, action, next_state, reward)

        # Move to the next state
        current_state = next_state

        # Turn policy_net back to training mode to optimize on a batch of experiences
        policy_net.train()

        # Perform one step of the optimization (on the target network) and record the loss value
        loss = optimize_model(batch_size, memory, policy_net, target_net, optimizer,
                              GAMMA=GAMMA, device=device)
        if loss is not None:
            running_loss += loss

        if done:
            # Save and print episode stats (duration and episode loss)
            episode_rewards.append(running_reward)
            mean_reward = (sum(episode_rewards[-100:]) / 100) if len(episode_rewards)>=100 else 0
            episode_loss.append(running_loss / (t + 1))
            plot_durations(episode_rewards, episode_loss)

            print("Episode: %d Cumulative Rewards: %d Episode Loss: %f, past 100 episodes avg reward: %f"
                  % (i_episode + 1, t + 1, (running_loss / (t + 1)), mean_reward))
            # Check if the problem is solved
            #  CartPole standard: average reward for the past 100 episode above 195
            if mean_reward > 195:
                print("\n\n\t Problem Solved !!!\n\n\n")

            break
    i_episode += 1

    # Update the target network, copying all weights and biases in DQN
    if i_episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Every save_ckpt_interval, save a checkpoint according to current i_episode.
    # Note that we use i_episode + 1
    if (i_episode + 1) % save_ckpt_interval == 0:
        save_checkpoint(ckpt_dir, policy_net, target_net, optimizer, memory, i_episode + 1, episode_rewards, episode_loss)
