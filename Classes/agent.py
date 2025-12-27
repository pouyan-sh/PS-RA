import torch as T
import torch.nn.functional as F
from Classes.networks import ActorNetwork, CriticNetwork
import numpy as np


class Agent:
    """
    Multi-head actor, local critic, target nets, and local learning.
    """

    def __init__(self, cfg, agent_id, state_dim, B, n_mod):
        self.cfg = cfg
        self.id = agent_id
        self.state_dim = state_dim
        self.B = B
        self.n_mod = n_mod

        # Action lengths
        self.action_dim = B + B + 1 + B * n_mod   # mask(B)+power(B)+cr(1)+mod_logits

        # Build actor/critic
        self.actor = ActorNetwork(
            alpha=cfg.actor_lr,
            input_dims=state_dim,
            fc1_dims=cfg.actor_layers[0],
            fc2_dims=cfg.actor_layers[1],
            n_agents=cfg.U,
            n_actions=self.action_dim,
            name='actor',
            agent_label=agent_id,
            B=B,
            n_mod=n_mod,
            P_max=cfg.P_max
        )

        self.critic = CriticNetwork(
            beta=cfg.critic_lr,
            input_dims=state_dim,
            fc1_dims=cfg.local_critic_layers[0],
            fc2_dims=cfg.local_critic_layers[1],
            fc3_dims=cfg.local_critic_layers[2],
            n_agents=cfg.U,
            n_actions=self.action_dim,
            name='critic',
            agent_label=agent_id
        )

        # Targets
        self.target_actor = ActorNetwork(
            alpha=cfg.actor_lr,
            input_dims=state_dim,
            fc1_dims=cfg.actor_layers[0],
            fc2_dims=cfg.actor_layers[1],
            n_agents=cfg.U,
            n_actions=self.action_dim,
            name='target_actor',
            agent_label=agent_id,
            B=B,
            n_mod=n_mod,
            P_max=cfg.P_max
        )

        self.target_critic = CriticNetwork(
            beta=cfg.critic_lr,
            input_dims=state_dim,
            fc1_dims=cfg.local_critic_layers[0],
            fc2_dims=cfg.local_critic_layers[1],
            fc3_dims=cfg.local_critic_layers[2],
            n_agents=cfg.U,
            n_actions=self.action_dim,
            name='target_critic',
            agent_label=agent_id
        )

        self.update_network_parameters(tau=1.0)

    # -------------------------------------------------------
    def flatten_action(self, action_dict):
        """
        action_dict: {
            'mask': Tensor(B, B),
            'power': Tensor(B, B),
            'cr': Tensor(B, 1),
            'mod': Tensor(B, n_mod)
        }
        returns Tensor(B, action_dim)
        """
        return T.cat([
            action_dict['mask'],                 # (B)
            action_dict['power'],                # (B)
            action_dict['cr'],                   # (1)
            action_dict['mod'].reshape(action_dict['mod'].shape[0], -1)       # (B*n_mod)
        ], dim=1)
    

    # -------------------------------------------------------
    def flatten_action_tensor(self, a_tensor):
        """
        Input: raw actor output tensor (batch, action_dim)
        Output: flattened action tensor (batch, action_dim)
        (mask(B), power(B), cr, modulation_logits(n_mod))
        """
        return a_tensor   # actor already outputs flat vector

    # -------------------------------------------------------
    def choose_action(self, obs_tensor, noise_scale=0.0):
        self.actor.eval()
        # actor returns dict of tensors
        out = self.actor(obs_tensor.to(self.actor.device))
        mu = self.flatten_action(out)

        # apply noise to the flat tensor (not the dict)
        if noise_scale > 0:
           noise = T.randn_like(mu) * noise_scale
           mu = mu + noise

        self.actor.train()
        return mu.detach().cpu().numpy()

    # -------------------------------------------------------
    def local_learn(self, global_loss, state, action, reward_l, state_, done):
        device = self.critic.device

        s = state.to(device)
        s_ = state_.to(device)
        a = action.to(device)
        r = reward_l.to(device).unsqueeze(1)
        d = done.to(device).unsqueeze(1)


        self.target_actor.eval()
        self.target_critic.eval()


        # target critic
        with T.no_grad():
            a_target = self.target_actor(s_)
            a_target_flat = self.flatten_action(a_target)
            q_next = self.target_critic(s_, a_target_flat)
            q_next[d.bool()] = 0.0
            target = r + self.cfg.gamma * q_next

        self.critic.train()
        # critic update
        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, target.detach())
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # actor update
        self.actor.train()
        self.actor.optimizer.zero_grad()
        actor_out = self.actor(s)
        a_curr_flat = self.flatten_action(actor_out)
        actor_loss_local = -self.critic(s, a_curr_flat)
        # if global_loss.dim() == 1:
        #     global_loss = global_loss.view(-1, 1)
        # combined_loss = actor_loss_local + global_loss
        # actor_loss = combined_loss.mean()
        actor_loss = T.mean(actor_loss_local) + self.cfg.lambda_loss * T.mean(global_loss)

        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    # -------------------------------------------------------
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.cfg.tau

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
