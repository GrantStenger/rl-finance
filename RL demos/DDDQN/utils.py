from collections import namedtuple
import random
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import torch


# Experience Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Plot diagrams
# Create matplotlib figure and subplot axes
fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
fig.suptitle("Training stats")

def plot_durations(episode_rewards, episode_loss):
    global fig, ax1, ax2
    """Plot diagrams for episode durations and episode loss"""

    # Plot episode duration
    durations_t = torch.tensor(episode_rewards, dtype=torch.float)
    ax1.cla()
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Rewards')
    ax1.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) > 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        #plt.plot(means.numpy())
        ax1.plot(means.numpy())

    # Plot episode loss
    ax2.cla()
    ax2.set_title('Episode Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.plot(episode_loss)

    # Re-draw, show, and give the system a bit of time to display and refresh the window
    plt.draw()
    plt.show()
    plt.pause(0.01)

