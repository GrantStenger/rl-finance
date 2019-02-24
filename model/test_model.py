from model import Encoder, PolicyNet
import torch


################## Hyperparameters ##################
input_size = 4
hidden_size=128
num_layers = 2
dropout = 0.85
seq_len = 1024

state_size = 3
num_actions = 3
act_lim = 1

batch_size = 32

######################################################

encoder = Encoder(input_size, batch_size, hidden_size, num_layers, dropout)
policy_net = PolicyNet(state_size, num_actions, act_lim, batch_size, hidden_size)

print("Encoder network: ", encoder)
print("Policy network: ", policy_net)
print()

# Test encoder
test_input = torch.randn((batch_size, seq_len, input_size))
print("test_input shape: ", test_input.shape)

encoding = encoder(test_input)
print("encoding shape: ", encoding.shape)


# Test Policy Net
# One step forward propagation
state = torch.randn((batch_size, state_size))
decision, actions, log_prob = policy_net(state, encoding=encoding)

print("decision shape: ", decision.shape)
print("actions shape: ", actions.shape)
print("actions log prob shape: ", log_prob.shape)