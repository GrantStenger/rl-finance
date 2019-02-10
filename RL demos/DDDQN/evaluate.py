from model_DDDQN import select_action_for_evaluation, load_checkpoint
import gym
import torch
from itertools import count

# IMPORTANT: Set value for i_episode to indicate which checkpoint you want to use
#   for evaluation. 
i_episode = 400
ckpt_dir = "DDDQN_CartPoleV1_obs_checkpoints/"
input_size = 4
output_size = 2

# Make environment
env = gym.make('CartPole-v1')

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read checkpoint
policy_net, _, _, _, _, _, _ = load_checkpoint(ckpt_dir, i_episode, input_size, output_size, device=device)

# Turn the policy network into evaluation mode
policy_net.eval()

# Initialize the environment and state
observation = env.reset()
current_state = torch.tensor([observation], device=device, dtype=torch.float32)

for t in count():
    # Select and perform an action
    action = select_action_for_evaluation(current_state, policy_net)
    observation, reward, done, _ = env.step(action.item())
    env.render()
    reward = torch.tensor([reward], device=device)

    if not done:
        next_state = torch.tensor([observation], device=device, dtype=torch.float32)
    else:
        next_state = None

    # Move to the next state
    current_state = next_state

    if done:
        print("Episode: %d Cumulative Rewards: %d" % (i_episode + 1, t + 1))
        break