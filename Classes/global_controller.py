import torch as T
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from Classes.g_network import G_CriticNetwork


class GlobalController:
    """
    Global controller managing twin global critics and coordinating global actor updates.
    Compatible with G_CriticNetwork(state, action).
    """

    def __init__(self, cfg, input_state_dim, n_actions, n_agents):
        self.cfg = cfg
        self.n_agents = n_agents
        self.n_actions = n_actions

        # Dimension of concatenated global state — NOT including actions
        self.global_state_dim = input_state_dim * n_agents
        # joint action vector length across agents
        self.global_action_dim = n_actions * n_agents

        # ---------------------------
        # Build global critics
        # ---------------------------
        # Critic takes (state, action) as SEPARATE inputs
        self.critic1 = G_CriticNetwork(
            beta=cfg.critic_lr,
            input_dims=self.global_state_dim,
            fc1_dims=cfg.global_critic_layers[0],
            fc2_dims=cfg.global_critic_layers[1],
            fc3_dims=cfg.global_critic_layers[2],
            n_agents=n_agents,
            n_actions=self.global_action_dim,
            name='global_critic1',
            agent_label='gc'
        )

        self.critic2 = G_CriticNetwork(
            beta=cfg.critic_lr,
            input_dims=self.global_state_dim,
            fc1_dims=cfg.global_critic_layers[0],
            fc2_dims=cfg.global_critic_layers[1],
            fc3_dims=cfg.global_critic_layers[2],
            n_agents=n_agents,
            n_actions=self.global_action_dim,
            name='global_critic2',
            agent_label='gc'
        )

        # Target critics
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)

        self.learn_step = 0

    # ---------------------------
    # Global learning
    # ---------------------------
    def global_learn(self, agents_list, batch_states, batch_actions, batch_r_g, batch_r_l, batch_states_, batch_dones):
        device = self.critic1.device

        states = T.tensor(batch_states, dtype=T.float32).to(device)         # (B, global_state_dim)
        actions = T.tensor(batch_actions, dtype=T.float32).to(device)       # (B, global_action_dim)
        states_ = T.tensor(batch_states_, dtype=T.float32).to(device)       # (B, global_state_dim)

        rewards_g = T.tensor(batch_r_g, dtype=T.float32).unsqueeze(1).to(device)
        dones = T.tensor(batch_dones, dtype=T.float32).unsqueeze(1).to(device)

        B = states.shape[0]
        state_per_agent = self.global_state_dim // self.n_agents

        # -------------------------------------------------------
        # Compute next joint action (target actors)
        # -------------------------------------------------------
        with T.no_grad():
            action_list = []
            for i in range(self.n_agents):
                s_i = states_[:, i * state_per_agent: (i + 1) * state_per_agent]
                out = agents_list[i].target_actor(s_i)
                a_i_flat = agents_list[i].flatten_action(out)
                action_list.append(a_i_flat)

            target_actions = T.cat(action_list, dim=1)  # (B, global_action_dim)

            # Policy smoothing
            # n_mod = len(self.agents_networks[0].cfg.modulations)
            action_dim_per_agent = self.n_actions               # = mask(B)+power(B)+cr+mod_logits
            n_mod = len(self.cfg.modulations)
            B_agent = (action_dim_per_agent - 1 - n_mod) // 2
            noisy_parts = []
            for i in range(self.n_agents):
                offs = i * action_dim_per_agent
                mask = target_actions[:, offs : offs + B_agent]
                power = target_actions[:, offs + B_agent : offs + 2*B_agent]
                cr = target_actions[:, offs + 2*B_agent : offs + 2*B_agent + 1]
                mod_logits = target_actions[:, offs + 2*B_agent + 1 : offs + action_dim_per_agent]
                mask_noisy = T.clamp(mask + T.randn_like(mask) * 0.02, 0.0, 1.0)
                power_noisy = T.clamp(power + T.randn_like(power) * 0.1, min=0.0)
                cr_noisy = T.clamp(
                    cr + T.randn_like(cr) * 0.02,
                    self.cfg.CR_min,
                    self.cfg.CR_max
                )
                mod_noisy = mod_logits + T.randn_like(mod_logits) * 0.05
                noisy_parts.append(T.cat(
                   [mask_noisy, power_noisy, cr_noisy, mod_noisy],
                   dim=1
                ))
            target_actions_noisy = T.cat(noisy_parts, dim=1)

            # Critic input is (state, action), not concatenated
            q1_next = self.critic1_target(states_, target_actions_noisy)
            q2_next = self.critic2_target(states_, target_actions_noisy)
            q_next = T.min(q1_next, q2_next)
            q_next[dones.bool()] = 0.0

            target_Q = rewards_g + self.cfg.gamma * q_next

        # -------------------------------------------------------
        # Current Q
        # -------------------------------------------------------
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        loss_q1 = F.mse_loss(q1, target_Q.detach())
        loss_q2 = F.mse_loss(q2, target_Q.detach())
        critic_loss = loss_q1 + loss_q2

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        # Soft update
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        self.learn_step += 1
        if self.learn_step % self.cfg.policy_delay != 0:
            return

        # -------------------------------------------------------
        # Compute deterministic actions of CURRENT actors
        # -------------------------------------------------------
        action_pred_list = []
        for i in range(self.n_agents):
            s_i = states[:, i * state_per_agent: (i + 1) * state_per_agent]
            out = agents_list[i].actor(s_i)
            a_i_flat = agents_list[i].flatten_action(out)
            action_pred_list.append(a_i_flat)

        actions_pred = T.cat(action_pred_list, dim=1)

        # Global actor loss: -Q
        actor_global_loss = -self.critic1(states, actions_pred)   # shape = (B, 1)
        actor_global_loss = actor_global_loss.detach()


        # -------------------------------------------------------
        # Local updates of each agent (providing the global loss)
        # -------------------------------------------------------
        batch_r_l_t = T.tensor(batch_r_l, dtype=T.float32).to(device)

        for i in range(self.n_agents):

            s_local = states[:, i * state_per_agent:(i + 1) * state_per_agent]
            s_local_ = states_[:, i * state_per_agent:(i + 1) * state_per_agent]

            a_local = actions[:, i * self.n_actions:(i + 1) * self.n_actions]

            r_local = batch_r_l_t[:, i]

            agents_list[i].local_learn(
                global_loss=actor_global_loss,
                state=s_local,
                action=a_local,
                reward_l=r_local,
                state_=s_local_,
                done=dones.squeeze(1)
            )

    # ---------------------------
    # Soft update
    # ---------------------------
    def _soft_update(self, net, target_net):
        tau = self.cfg.tau
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
