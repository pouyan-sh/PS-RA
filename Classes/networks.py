"""
Classes/networks.py
ActorNetwork (multi-headed) and CriticNetwork (local) adapted & expanded.
"""

import os
import torch as T
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class ActorNetwork(nn.Module):
    """
    Multi-headed actor:
     - mask_head -> B outputs in [0,1] (sigmoid)
     - power_head -> B outputs >=0 (softplus)
     - cr_head -> scalar in [CR_min, CR_max] (sigmoid scaled)
     - mod_head -> logits length n_mod
    """
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_agents, n_actions,
                 name, agent_label, B, n_mod, chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        # naming / checkpoint
        self.name = name + '_' + str(agent_label)
        self.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '.pt')


        # layers
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        # heads
        self.mask_head = nn.Linear(self.fc2_dims, B)
        self.power_head = nn.Linear(self.fc2_dims, B)
        self.cr_head = nn.Linear(self.fc2_dims, 1)
        self.mod_head = nn.Linear(self.fc2_dims, n_mod)

        
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)


        f_out = 0.003

        self.mask_head.weight.data.uniform_(-f_out, f_out)
        self.mask_head.bias.data.uniform_(-f_out, f_out)

        self.power_head.weight.data.uniform_(-f_out, f_out)
        self.power_head.bias.data.uniform_(-f_out, f_out)

        self.cr_head.weight.data.uniform_(-f_out, f_out)
        self.cr_head.bias.data.uniform_(-f_out, f_out)

        self.mod_head.weight.data.uniform_(-f_out, f_out)
        self.mod_head.bias.data.uniform_(-f_out, f_out)


        # init
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        mask = T.sigmoid(self.mask_head(x))            # [0,1]
        power_raw = F.softplus(self.power_head(x))     # >=0
        cr_raw = T.sigmoid(self.cr_head(x))            # [0,1]
        mod_logits = self.mod_head(x)                  # raw logits
        return {'mask': mask, 'power': power_raw, 'cr': cr_raw, 'mod': mod_logits}

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"... saving actor checkpoint to {self.checkpoint_file} ...")
        T.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self):
        if not os.path.isfile(self.checkpoint_file):
           print(f"... actor checkpoint not found at {self.checkpoint_file}, skipping load ...")
           return
        print(f"... loading actor checkpoint from {self.checkpoint_file} ...")
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class CriticNetwork(nn.Module):
    """
    Local critic network with 3 hidden layers (per request).
    Input: concatenated local observation + local action (flattened).
    """
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, n_agents, n_actions, name, agent_label,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name + '_' + str(agent_label)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        self.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '.pt')


        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.bn3 = nn.LayerNorm(self.fc3_dims)
        
        # action projection
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        self.q = nn.Linear(self.fc3_dims, 1)


        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 1. / np.sqrt(self.fc3.weight.data.size()[0])
        self.fc3.weight.data.uniform_(-f3, f3)
        self.fc3.bias.data.uniform_(-f3, f3)

        f4 = 0.003
        self.q.weight.data.uniform_(-f4, f4)
        self.q.bias.data.uniform_(-f4, f4)

        f5 = 1. / np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f5, f5)
        self.action_value.bias.data.uniform_(-f5, f5)


        # init and optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        action_proj = self.action_value(action)
        x = F.relu(x + action_proj)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        q = self.q(x)
        return q
    
    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"... saving critic checkpoint to {self.checkpoint_file} ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if not os.path.isfile(self.checkpoint_file):
            print(f"... critic checkpoint not found at {self.checkpoint_file}, skipping load ...")
            return
        print(f"... loading critic checkpoint from {self.checkpoint_file} ...")
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

