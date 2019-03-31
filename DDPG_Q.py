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
    
class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(QNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim,256)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(256,128)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        self.fca1 = nn.Linear(action_dim,128)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.V1 = nn.Linear(128,128)
        self.V1.weight.data = fanin_init(self.V1.weight.data.size())

        self.V2 = nn.Linear(128,1)
        self.V2.weight.data.uniform_(-EPS,EPS)

        self.A1 = nn.Linear(128+128,128)
        self.A1.weight.data = fanin_init(self.A1.weight.data.size())

        self.A2 = nn.Linear(128,1)
        self.A2.weight.data.uniform_(-EPS,EPS)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        
        V = F.relu(self.V1(s2))
        V = self.V2(V)

        x = torch.cat((s2,a1),dim=1)
        A = F.relu(self.A1(x))
        A = self.A2(A)
        
        return A, V*V