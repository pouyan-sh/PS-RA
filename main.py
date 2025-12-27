import os
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from config import cfg
from Classes.buffer import ReplayBuffer
from Classes.agent import Agent
from Classes.global_controller import GlobalController
from environment import OFDMAEnv
import time


def flatten_obs(obs, B):
    arr = []
    arr.append(float(obs['alfa']))
    arr.append(float(obs['delay']))
    arr.append(float(obs['SSM']))
    arr.append(float(obs['cr']))
    arr.append(float(obs['P_max']))
    arr.append(float(obs['BER_th']))
    arr.append(float(obs['Phis']))
    arr.append(float(obs['avg_weigth']))
    arr.append(float(obs['num_images']))
    arr.extend(list(obs['rb_gains']))
    arr.extend(list(obs['rb_interference']))
    return np.array(arr, dtype=np.float32)


def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = OFDMAEnv(cfg)
    obs = env.reset()

    U = cfg.U
    B = cfg.B
    n_mod = len(cfg.modulations)

    sample = flatten_obs(obs[0], B)
    state_dim = sample.shape[0]

    agents = [Agent(cfg, agent_id=i, state_dim=state_dim, B=B, n_mod=n_mod)
              for i in range(U)]

    action_dim = B + B + 1 + n_mod

    buffer = ReplayBuffer(cfg.replay_buffer_size, input_shape=state_dim, n_actions=action_dim, n_agents=U)

    global_controller = GlobalController(cfg, input_state_dim=state_dim,
                                         n_actions=action_dim, n_agents=U)

    ep_rewards = []

    for ep in range(cfg.episodes):
        obs = env.reset()
        state_global = np.concatenate([flatten_obs(o, B) for o in obs])

        ep_reward = 0

        for t in range(cfg.episode_length):
            actions = []

            for i in range(U):
                obs_i = flatten_obs(obs[i], B)
                obs_t = T.tensor(obs_i).unsqueeze(0)
                a = agents[i].choose_action(obs_t, noise_scale=cfg.gaussian_noise_sigma)
                actions.append(a[0])

            next_obs, r_local, r_global, done, info = env.step(actions)
            next_state_global = np.concatenate([flatten_obs(o, B) for o in next_obs])

            buffer.store_transition(state_global.astype(np.float32),
                                    np.concatenate(actions).astype(np.float32),
                                    float(r_global),
                                    np.array(r_local).astype(np.float32),
                                    next_state_global.astype(np.float32),
                                    done)

            state_global = next_state_global
            obs = next_obs
            ep_reward += r_global

            if len(buffer) >= cfg.batch_size:
                s, a, rg, rl, s_, d = buffer.sample_buffer(cfg.batch_size)
                global_controller.global_learn(agents, s, a, rg, rl, s_, d)

            if done:
                break

        ep_rewards.append(ep_reward)
        print(f"[Episode {ep+1}/{cfg.episodes}]  Global Reward = {ep_reward:.3f}")

    plt.plot(ep_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Global Reward")
    plt.grid(True)
    plt.savefig("logs/ep_rewards.png")
    plt.show()


if __name__ == "__main__":
    main()
