from save_and_load import load_checkpoint
import gym
import torch
from itertools import count

# IMPORTANT: Set value for i_episode to indicate which checkpoint you want to use
#   for evaluation.
i_epoch = 760
num_eval_episodes = 32      # Number of episodes to run for evaluation
env_name = "LunarLander-v2"
ckpt_dir = "simplePG_Adam_%s_obs_checkpoints/" % (env_name)

input_size = 8
output_size = 4
layer_sizes = [input_size, 128, 128, 128, output_size]

# Make environment
env = gym.make(env_name)

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read checkpoint
policy_net, _, _ = load_checkpoint(ckpt_dir, i_epoch, layer_sizes, device=device)


# Record evaluation info
evaluation_info = {"episode rewards": [],
                   "episode durations": [],
                   "max reward acieved": 0,}

# Turn the policy network into evaluation mode
policy_net.eval()

for i_episode in range(num_eval_episodes):

    # Initialize the environment and state
    observation = env.reset()
    current_state = torch.tensor([observation], device=device, dtype=torch.float32)

    running_reward = 0
    for t in count():
        action = policy_net(current_state)

        # Interact with the environment
        observation, reward, done, _ = env.step(action.item())
        env.render()

        # Record reward
        running_reward += reward

        if not done:
            current_state = torch.tensor([observation], device=device, dtype=torch.float32)
        else:
            evaluation_info["episode rewards"].append(running_reward)
            if evaluation_info["max reward acieved"] < running_reward:
                evaluation_info["max reward acieved"] = running_reward
            evaluation_info["episode durations"].append(t + 1)

            # Print episode stats
            print("\n============= Episode: %d =============" % (i_episode + 1))
            print("Episode rewards: %f " % (running_reward))
            print("Episode durations: %f " % (t + 1))
            break

# Print evaluation result
print ("\n\n################ Evaluation Result ################")
print("\tMax reward achieved: %f " % (evaluation_info["max reward acieved"]))
print("\tPast %d episodes mean rewards: %f" %
      (num_eval_episodes, sum(evaluation_info["episode rewards"]) / (num_eval_episodes)))
print("\tpast %d episodes mean durations: %f" %
      (num_eval_episodes, sum(evaluation_info["episode durations"]) / (num_eval_episodes)))

