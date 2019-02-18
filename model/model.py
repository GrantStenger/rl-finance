import torch
import torch.nn as nn
from torch.nn import LSTM, LSTMCell, Linear
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions import Normal


class Encoder(nn.Module):

    def __init__(self, input_size, batch_size=1, hidden_size=128, num_layers=2, dropout=0.85):
        """ Construct a multilayer LSTM that computes the encoding vector"""

        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.LSTM = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                         dropout=dropout)          # encoding size is the same as the hidden size

    def forward(self, input, h_0=None, c_0=None):
        """ input should have size (batch_size, seq_len, input_size)"""

        # Forward propagation to calculate the encoding vector
        # We only care about the final unit output, i.e., the hidden state of the final unit
        if (h_0 is not None) and (c_0 is not None):
            _, (h_n, _) = self.LSTM(input, (h_0, c_0))
        else:
            _, (h_n, _) = self.LSTM(input)
        # h_n should have shape (num_layers, batch_size, hidden_size). We only want the content from the last layer
        encoding = h_n[-1, :, :]
        # squeeze the encoding vector so that it has shape (batch_size, hidden_size)
        encoding = encoding.squeeze(1)

        return encoding


class PolicyNet(nn.Module):

    def __init__(self, state_size, num_actions, act_lim=1, batch_size=1, hidden_size=128):
        """ Construct a multilayer LSTM that computes the action given the state

            - shape of input state is given by state_size
            - dimensions of the orthogonal action space is given by num_actions, whereas act_lim gives the numerical bound for action values
            - hidden_size should match that of the encoding network (i.e. the size of the encoding layer)

            The agent should first decide which dimension to act on and then decide the numerical value of the aciton on that dimension
        """
        super(PolicyNet, self).__init__()

        self.state_size = state_size
        self.num_actions = num_actions
        self.act_lim = act_lim
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.LSTMCell = LSTMCell(input_size=state_size, hidden_size=hidden_size)

        # Linear layer that decides the dimension the agent wants to act on.
        #   Return the logits to be used to construct a Categorical distribution
        self.FC_decision = Linear(hidden_size, num_actions)
        # Linear layer that computes the mean value of the agent's action on each dimension
        self.FC_values_mean = Linear(hidden_size, num_actions)
        # Linear layer that computes the log standard deviation of the agent's action on each dimension
        self.FC_values_logstd = Linear(hidden_size, num_actions)

    def forward(self, state, h_0=None, c_0=None):
        """
            - At the first step, h_0 should be the encoding vector from Encoder with shape (batch_size, hidden_size).
                Note that we should transfrom the shape properly.

        """


        # Forward propagation
        if h_0 is None:
            h_0 = torch.zeros((self.batch_size, self.hidden_size))
        if c_0 is None:
            c_0 = torch.zeros((self.batch_size, self.hidden_size))

        h_1, c_1 = self.LSTMCell(state, (h_0, c_0))

        decision_logit = self.FC_decision(h_1)
        values_mean = self.FC_values_mean(h_1)
        values_logstd = self.FC_values_logstd(h_1)

        # Take the exponentials of log standard deviation
        values_std = torch.exp(values_logstd)

        # Create a categorical (multinomial) distribution from which we can sample a decision on the action dimension
        m_decision = OneHotCategorical(logits=decision_logit)

        # Sample a decision and calculate its log probability. decision of shape (num_actions,)
        decision = m_decision.sample()
        decision_log_prob = m_decision.log_prob(decision)

        # Create a list of Normal distributions for sampling actions in each dimension
        m_values = []
        action_values = None
        actions_log_prob = None
        for i in range(self.num_actions):
            m_values.append(Normal(values_mean[:, i], values_std[:, i]))
            if action_values is None:
                action_values = m_values[-1].sample().unsqueeze(1)                    # Unsqueeze to spare the batch dimension
                actions_log_prob = m_values[-1].log_prob(action_values[:, -1]).unsqueeze(1)
            else:
                action_values = torch.cat([action_values, m_values[-1].sample().unsqueeze(1)], dim=1)
                actions_log_prob = torch.cat([actions_log_prob, m_values[-1].log_prob(action_values[:, -1]).unsqueeze(1)], dim=1)

        # Filter the final action value in the intended action dimension
        final_action_values = (action_values * decision).sum(dim=1)
        final_action_log_prob = (actions_log_prob * decision).sum(dim=1)

        # Calculate the final log probability
        #   Pr(action value in the ith dimension) = Pr(action value given the agent chooses the ith dimension)
        #                                           * Pr(the agent chooses the ith dimension
        log_prob = decision_log_prob + final_action_log_prob

        # Return the hidden and cell states as well in order to pass in the LSTM cell in the next time step
        return decision, final_action_values, log_prob, h_1, c_1


