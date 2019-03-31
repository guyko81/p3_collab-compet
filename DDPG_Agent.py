import numpy as np
import random
from collections import namedtuple, deque

from DDPG_P import Policy
from DDPG_Q import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

import random


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-4               # learning rate 
UPDATE_EVERY = 1        # how often to update the network
UPDATE_EVERY2 = 1
LR2 = 1e-4
num_of_batch_step = 1

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, n_agents, seed, action_lim=1, activation=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.embedding_size = 16

        self.action_lim = action_lim

        self.activation = activation

        # Q-Network
        self.policy_local = Policy(state_size, action_size).to(device)
        self.policy_target = Policy(state_size, action_size).to(device)
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        
        self.policy_optimizer = optim.Adam(self.policy_local.parameters(), lr=LR2)
        self.qnetwork_optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR2)
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.t_step2 = 0
        
        self.expected = torch.from_numpy(np.asarray([[0]]))
        self.target = torch.from_numpy(np.asarray([[0]]))
        self.local = torch.from_numpy(np.asarray([[0]]))
        
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_size)

    def _activation(self, x, activation=None, numpy=True):
        if numpy:
            if activation is None:
                t = x
            if activation == 'tanh':
                t = np.tanh(x)
            if activation == 'sigmoid':
                t = 1/(1+np.exp(-x))

        if not numpy:
            if activation is None:
                t = x
            if activation == 'tanh':
                t = torch.tanh(x)
            if activation == 'sigmoid':
                t = torch.sigmoid(x)
        
        return t * self.action_lim 

    def step(self, states, actions, rewards, next_states, dones):
        
        # Save experience in replay memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
 
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                for b in range(num_of_batch_step):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_local.eval()
        with torch.no_grad():
            action_policy, _ = self.policy_local(state)
        self.policy_local.train()
        self.qnetwork_local.train()

        action_policy = action_policy.data.cpu().numpy().astype(float).reshape(-1,)
        if eps < np.random.random():
            # + action_policy*self.noise.sample()
            return self._activation(action_policy, activation=self.activation, numpy=True)
        else:
            return self._activation(np.random.normal(size=self.n_agents*self.action_size), activation=self.activation, numpy=True)
        
    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        self.qnetwork_target.eval()
        self.policy_target.eval()


        #########################################################
        ########## Q-Network Optimization #######################
        #########################################################

        Expected_actions, _ = self.policy_target(next_states)
        Expected_actions = self._activation(Expected_actions, activation=self.activation, numpy=False)
        Q_targets_next, _ = self.qnetwork_target(next_states, Expected_actions)
        Q_targets_next = Q_targets_next.detach()
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected_mu, _ = self.qnetwork_local(states, actions)

        loss = F.smooth_l1_loss(Q_expected_mu, Q_targets)

        self.qnetwork_optimizer.zero_grad()
        loss.backward()
        self.qnetwork_optimizer.step()


        #########################################################
        ########## End of Q-Network Optimization ################
        #########################################################



        #########################################################
        ########## Policy-Network Optimization  - Expected Q ####
        #########################################################

        policy, _ = self.policy_local(states)
        policy = self._activation(policy, activation=self.activation, numpy=False)
        Q_local_mu, _ = self.qnetwork_local(states, policy)
        P_expexted_loss = -1*torch.mean(Q_local_mu)
        self.policy_optimizer.zero_grad()
        P_expexted_loss.backward()
        self.policy_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        self.soft_update(self.policy_local, self.policy_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)