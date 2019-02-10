from model_DDDQN import load_checkpoint
from utils import plot_durations
import matplotlib.pyplot as plt
import torch

# IMPORTANT: Set value for i_episode to indicate which checkpoint you want to use
#   for evaluation.
i_episode = 400
ckpt_dir = "DDDQN_CartPoleV1_obs_checkpoints/"
input_size = 4
output_size = 2

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read checkpoint
policy_net, _, _, _, _, episode_rewards, episode_loss = \
    load_checkpoint(ckpt_dir, i_episode, input_size, output_size, device=device)

# Plot figure
plot_durations(episode_rewards, episode_loss)