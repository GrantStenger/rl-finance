from save_and_load import load_checkpoint
from utils import plot_durations
import matplotlib.pyplot as plt
import torch


# IMPORTANT: Set value for i_episode to indicate which checkpoint you want to use
#   for evaluation.
i_epoch = 650
start_idx = 0
end_idx = i_epoch

input_size = 8
output_size = 4
layer_sizes = [input_size, 128, 128, 128, output_size]        # The MLP network architecture

env_name = "LunarLander-v2"
ckpt_dir = "simplePG_Adam_%s_obs_checkpoints/" % (env_name)

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read checkpoint
_, _, training_info = \
    load_checkpoint(ckpt_dir, i_epoch, layer_sizes, device=device)

# Plot figure
plot_durations(training_info["epoch mean rewards"],
               (start_idx, end_idx))