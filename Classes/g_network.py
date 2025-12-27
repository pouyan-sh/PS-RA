import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class G_CriticNetwork(nn.Module):
    """
    Global critic with 3 hidden layers.
    Expects separate inputs: forward(state, action)
    """

    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims,
                 n_agents, n_actions, name, agent_label,
                 chkpt_dir='tmp/ddpg'):

        super(G_CriticNetwork, self).__init__()

        self.state_dim = input_dims
        self.action_dim = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        self.name = f"{name}_{agent_label}"

        self.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_gcritic.pt')


        # Layers
        self.fc1 = nn.Linear(self.state_dim, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        # Action → projection
        self.action_value = nn.Linear(self.action_dim, self.fc2_dims)

        # Final Q
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


        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    # ---------------------------
    def forward(self, state, action):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        a_proj = self.action_value(action)
        x = F.relu(x + a_proj)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)

        return self.q(x)
    
    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"... saving gcritic checkpoint to {self.checkpoint_file} ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if not os.path.isfile(self.checkpoint_file):
            print(f"... gcritic checkpoint not found at {self.checkpoint_file}, skipping load ...")
            return
        print(f"... loading gcritic checkpoint from {self.checkpoint_file} ...")
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

    def save_best(self):
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_best.pt')
        print(f"... saving BEST gcritic to {checkpoint_file} ...")
        T.save(self.state_dict(), checkpoint_file)
