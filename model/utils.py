import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import torch


# Plot diagrams
# Create matplotlib figure and subplot axes
fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
fig.suptitle("Training stats")

def plot_durations(episode_rewards, idx_range=None, mean_interval=5):
    """Plot diagrams for episode durations and episode loss"""
    global fig, ax1, ax2, ax3

    if idx_range is not None:
        start_idx, end_idx = idx_range
        x_axis = range(start_idx, end_idx)

    # Plot episode duration
    durations_t = torch.tensor(episode_rewards, dtype=torch.float)
    ax1.cla()
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Rewards')
    if idx_range is None:
        ax1.plot(durations_t.numpy())
    else:
        ax1.plot(x_axis, durations_t.numpy()[start_idx : end_idx])

    # Take 100 episode averages and plot them too
    if len(durations_t) > mean_interval:
        means = durations_t.unfold(0, mean_interval, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(mean_interval - 1), means))
        if idx_range is None:
            ax1.plot(means.numpy())
        else:
            ax1.plot(x_axis, means.numpy()[start_idx : end_idx])

    # # Plot minibatch loss
    # ax2.cla()
    # ax2.set_title('Minibatch Loss')
    # ax2.set_xlabel('Episode')
    # ax2.set_ylabel('Minibatch loss')
    # if idx_range is None:
    #     ax2.plot(minibatch_loss)
    # else:
    #     ax2.plot(x_axis, minibatch_loss[start_idx : end_idx])
    #
    # # Plot episode loss
    # ax3.cla()
    # ax3.set_title('Episode Loss')
    # ax3.set_xlabel('Episode')
    # ax2.set_ylabel('Episode loss')
    # if idx_range is None:
    #     ax3.plot(episode_loss)
    # else:
    #     ax3.plot(x_axis, episode_loss[start_idx : end_idx])

    # Re-draw, show, and give the system a bit of time to display and refresh the window
    plt.draw()
    plt.show()
    plt.pause(0.01)

