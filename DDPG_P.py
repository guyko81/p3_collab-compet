import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

def logrelu(x):
    return torch.log(x*x+1)*x

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)
    
class Policy(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :return:
        """
        super(Policy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim,256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.mu1 = nn.Linear(128,64)
        self.mu1.weight.data = fanin_init(self.mu1.weight.data.size())

        self.mu2 = nn.Linear(64,action_dim)
        self.mu2.weight.data.uniform_(-EPS,EPS)

        self.sigma1 = nn.Linear(128,64)
        self.sigma1.weight.data = fanin_init(self.sigma1.weight.data.size())

        self.sigma2 = nn.Linear(64,action_dim)
        self.sigma2.weight.data.uniform_(-EPS,EPS)

    def forward(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        x = F.relu(self.fc1(state))
        #x = F.dropout(x, 0.1)
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, 0.1)

        mu = F.relu(self.mu1(x))
        mu = self.mu2(mu)

        sigma = F.relu(self.sigma1(x))
        sigma = self.sigma2(sigma)
        sigma = torch.clamp(sigma, -1, 1)

        return mu, sigma*sigma