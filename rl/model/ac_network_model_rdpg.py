import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x, h0=None):
        if len(x.size()) == 3:
            # TimeDistributed + Dense
            # shape: (seq_len, batch, features)
            seq_len, b, n = x.size(0), x.size(1), x.size(2)
            # merge batch and seq dimensions
            x_reshape = x.contiguous().view(seq_len * b, n)

        elif len(x.size()) == 4:
            # TimeDistributed + RNN
            # shape: (seq_len, batch, agents, features)
            seq_len, b, n, d = x.size(0), x.size(1), x.size(2), x.size(3)
            # merge batch and seq dimensions
            x_reshape = x.contiguous().view(seq_len, b * n, d)

            # return
        if isinstance(self.module, nn.Linear):
            # TimeDistributed + Dense
            y = self.module(x_reshape)
            # We have to reshape Y
            # shape: (seq_len, batch, documents, features)
            y = y.contiguous().view(seq_len, b, n, y.size()[-1])
            return y

        elif isinstance(self.module, nn.LSTM):
            # TimeDistributed + RNN
            y, (h, c) = self.module(x_reshape, h0)
            x_reshape.size()
            # shape: (seq_len, batch x agents, features)
            y = y.contiguous().view(seq_len, b, n, y.size()[-1])
            # shape: (seq_len, batch, agents, features)

            # shape: (batch, agents, features)
            h = h.contiguous().view(h.size()[0], b, n, h.size()[-1])
            c = c.contiguous().view(c.size()[0], b, n, c.size()[-1])
            return y, (h, c)

        elif isinstance(self.module, nn.GRU):
            # TimeDistributed + RNN
            y, h = self.module(x_reshape, h0)
            # shape: (seq_len, batch x agents, features)
            y = y.contiguous().view(seq_len, b, n, y.size()[-1])
            # shape: (seq_len, batch, agents, features)

            # shape: (batch, agents, features)
            h = h.contiguous().view(1, b, n, h.size()[-1])
            return y, h

        else:
            raise ImportError('Not Supported Layers!')


class ActorNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, input_dim, out_dim):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int) : Number of dimensions in input  (agents, observation)
            out_dim (int)   : Number of dimensions in output
            hidden_dim (int) : Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(ActorNetwork, self).__init__()

        self.nonlin = F.relu
        self.dense1 = nn.Linear(input_dim, 64)
        self.lstmTime = TimeDistributed(nn.LSTM(64, 64, num_layers=1, bidirectional=False))
        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.bilstmAgent = TimeDistributed(nn.LSTM(64, 32, num_layers=1, bidirectional=True))
        self.dense2 = nn.Linear(64, out_dim)
        self.dense3 = nn.Linear(64, input_dim)

    def forward(self, obs):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): policy, next_state
        """
        # obs: (time, batch, agent, feature)
        hid = F.relu(self.dense1(obs))
        hid, hcTime = self.lstmTime(hid, h0=None)
        hid = hid.permute(2, 1, 0, 3)
        hid, hcAgent = self.bilstmAgent(hid, h0=None)
        hid = hid.permute(2, 1, 0, 3)
        hid = F.relu(hid)
        policy = self.dense2(hid)
        policy = nn.Softmax(dim=-1)(policy)
        next_state = self.dense3(hid)
        return policy, next_state


class CriticNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, input_dim, out_dim):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int): Number of dimensions in input  (agents, observation)
            out_dim (int): Number of dimensions in output

            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(CriticNetwork, self).__init__()

        self.nonlin = F.relu
        self.dense1 = TimeDistributed(nn.Linear(input_dim, 64))
        self.lstmTime = nn.LSTM(64, 64, num_layers=1, bidirectional=False)
        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.lstmAgent = nn.LSTM(64, 64, num_layers=1, bidirectional=False)
        self.dense2 = nn.Linear(64, out_dim)
        self.dense3 = nn.Linear(64, out_dim)

    def forward(self, obs, action):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Q-function
            out (PyTorch Matrix): reward
        """
        obs_act = torch.cat((obs, action), dim=-1)
        hid = F.relu(self.dense1(obs_act))
        hid, _ = self.lstmTime(hid, None)
        hid, _ = self.lstmAgent(hid, None)
        hid = F.relu(hid[:, -1, :])
        Q = self.dense2(hid)
        r = self.dense3(hid)
        return Q, r

if __name__ == '__main__':
    actor = ActorNetwork(input_dim=10, out_dim=5)
    critic = CriticNetwork(input_dim=10, out_dim=1)

    s = torch.randn(15, 32, 3, 10)
    pred_actor = actor.forward(s)
    pred_actor[0].size()
    pred_actor[1].size()

    a = torch.randn(15, 32, 3, 5)
    pred_critic = critic.forward(s, a)