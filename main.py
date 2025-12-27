import csv
import random
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
import shutil

RUN_ID = f"seed{cfg.seed}_" + time.strftime("%Y%m%d_%H%M%S")

if T.cuda.is_available():
    if cfg.use_deterministic:
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.deterministic = True
    else:
        T.backends.cudnn.benchmark = True
        T.backends.cudnn.deterministic = False

def flatten_obs(obs, B):
    arr = []
    arr.append(obs['task'])
    arr.append(float(obs['delay']))
    arr.append(float(obs['alfa']))
    arr.append(float(obs['SSM']))
    arr.append(float(obs['cr']))
    arr.append(float(obs['P_max']))
    arr.append(float(obs['BER_th']))
    arr.append(float(obs['Phis']))
    arr.append(float(obs['avg_weigth']))
    arr.append(float(obs['num_images']))
    arr.extend(list(obs['rb_gains']))
    arr.append(float(obs['delay_violation']))
    arr.extend(list(obs['rb_interference']))
    arr.extend(list(obs['rb_mask_prev']))    
    arr.extend(list(obs['rb_power_prev']))
    arr.extend(list(obs['sinr_prev']))
    arr.append(float(obs['rate_prev']))
    arr.append(float(obs['num_active_rb']))
    arr.extend(list(obs['rb_load']))
    arr.append(float(obs['num_collided_rb']))
    arr.append(float(obs['power_usage']))

    return np.array(arr, dtype=np.float32)


def train():
    os.makedirs("checkpoints", exist_ok=True)
    run_dir = f"logs/{RUN_ID}"
    os.makedirs(run_dir, exist_ok=True)

    # Reproducibility
    np.random.seed(cfg.seed)
    T.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # Save config
    with open(f"{run_dir}/config.txt", "w") as f:
        for k, v in vars(cfg).items():
            f.write(f"{k}: {v}\n")

    # -------------------------------
    # Hyperparameter summary
    # -------------------------------
    hparam_file = "logs/hparam_results.csv"

    if not os.path.exists(hparam_file):
       with open(hparam_file, "w") as f:
            f.write("run_id,actor_lr,critic_lr,gamma,tau\n")

    with open(hparam_file, "a") as f:
        f.write(
            f"{RUN_ID},"
            f"{cfg.actor_lr},"
            f"{cfg.critic_lr},"
            f"{cfg.gamma},"
            f"{cfg.tau}\n"
        )


    env = OFDMAEnv(cfg)
    obs = env.reset()

    U = cfg.U
    B = cfg.B
    n_mod = len(cfg.modulations)

    sample = flatten_obs(obs[0], B)
    state_dim = sample.shape[0]

    agents = [Agent(cfg, agent_id=i, state_dim=state_dim, B=B, n_mod=n_mod)
              for i in range(U)]

    action_dim = B + B + 1 + B * n_mod

    buffer = ReplayBuffer(cfg.replay_buffer_size, input_shape=state_dim, n_actions=action_dim, n_agents=U)

    global_controller = GlobalController(cfg, input_state_dim=state_dim,
                                         n_actions=action_dim, n_agents=U)

    ep_rewards = []
    log_file = open(f"{run_dir}/train_log.csv", "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "episode",
        "avg_global_reward",
        "steps",
        "avg_local_reward",
        "avg_sum_rate",
        "avg_collisions",
        "avg_ssm",
        "avg_delay",
        "satisfy_fraction",
        "buffer_size"
    ])

    best_global_reward = -np.inf


    # ---- episode-level logs ----
    ep_global_rewards = []
    ep_avg_local_rewards = []
    ep_avg_ssm = []
    ep_avg_delay = []
    ep_satisfy_frac = []
    ep_avg_sinr_db = []

    for ep in range(cfg.episodes):
        sum_rates_ep = []
        collisions_ep = []
        obs = env.reset()
        state_global = np.concatenate([flatten_obs(o, B) for o in obs])

        ep_reward = 0
        step_count = 0
        ep_local_rewards = []
        ep_ssm = []
        ep_sinr = []
        ep_delay = []
        ep_satisfy = []

        for t in range(cfg.episode_length):
            actions = []

            for i in range(U):
                obs_i = flatten_obs(obs[i], B)
                obs_t = T.tensor(obs_i).unsqueeze(0)
                a = agents[i].choose_action(obs_t, noise_scale=cfg.gaussian_noise_sigma)
                actions.append(a[0])

            next_obs, r_local, r_global, done, info = env.step(actions)
            step_count += 1

            ep_local_rewards.append(np.mean(r_local))
            ep_ssm.append(np.mean(info["SSM"]))
            ep_delay.append(info["avg_delay"])
            ep_satisfy.append(info["satisfy_fraction"])
            ep_sinr.append(np.mean(info["sinr_db"]))
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

            sum_rates_ep.append(np.sum(info["rates"]))
            collisions_ep.append(info.get("num_collided_rb", 0))


            if len(buffer) >= cfg.batch_size:
                s, a, rg, rl, s_, d = buffer.sample_buffer(cfg.batch_size)
                global_controller.global_learn(agents, s, a, rg, rl, s_, d)

            if done:
                break

        ep_reward_avg = ep_reward / max(step_count, 1)
        print(f"[Episode {ep+1}/{cfg.episodes}]  Avg Global Reward = {ep_reward_avg:.3f}  (steps={step_count})")
        ep_global_rewards.append(ep_reward_avg)
        ep_avg_local_rewards.append(np.mean(ep_local_rewards))
        ep_avg_ssm.append(np.mean(ep_ssm))
        ep_avg_delay.append(np.mean(ep_delay))
        ep_satisfy_frac.append(np.mean(ep_satisfy))

        if len(ep_sinr) > 0:
           ep_avg_sinr_db.append(np.mean(ep_sinr))
        else:
           ep_avg_sinr_db.append(0.0)

        # -------- Save last checkpoints --------
        if (ep + 1) % cfg.save_interval == 0 or ((ep + 1) == cfg.episodes):
            for agent in agents:
                agent.actor.save_checkpoint()
                agent.critic.save_checkpoint()
                agent.target_actor.save_checkpoint()
                agent.target_critic.save_checkpoint()
            global_controller.critic1.save_checkpoint()
            global_controller.critic2.save_checkpoint()
            global_controller.critic1_target.save_checkpoint()
            global_controller.critic2_target.save_checkpoint()


        # -------- Save best checkpoints --------
        if ep_reward > best_global_reward:
            best_global_reward = ep_reward
            print(">>> New BEST model found, saving...")

            for agent in agents:
                agent.actor.save_best()
                agent.critic.save_best()

            global_controller.critic1.save_best()
            global_controller.critic2.save_best()

        if (ep + 1) % 10 == 0:
           plt.figure(figsize=(14, 10))

           # ---- Global reward ----
           plt.subplot(2, 3, 1)
           plt.plot(ep_global_rewards)
           plt.title("Global Reward")
           plt.xlabel("Episode")
           plt.grid()

           # ---- Avg local reward ----
           plt.subplot(2, 3, 2)
           plt.plot(ep_avg_local_rewards)
           plt.title("Avg Local Reward")
           plt.grid()

           # ---- Avg SSM ----
           plt.subplot(2, 3, 3)
           plt.plot(ep_avg_ssm)
           plt.title("Avg SSM")
           plt.ylim(0, 1)
           plt.grid()

           # ---- Avg Delay ----
           plt.subplot(2, 3, 4)
           plt.plot(ep_avg_delay)
           plt.title("Avg Delay")
           plt.grid()

           # ---- Satisfy Fraction ----
           plt.subplot(2, 3, 5)
           plt.plot(ep_satisfy_frac)
           plt.title("Satisfy Fraction")
           plt.ylim(0, 1)
           plt.grid()

           # ---- Avg SINR (dB) ----
           plt.subplot(2, 3, 6)
           plt.plot(ep_avg_sinr_db)
           plt.title("Avg SINR (dB)")
           plt.grid()

           plt.tight_layout()
           plt.savefig(f"{run_dir}/train_progress_ep_{ep+1}.png")
           plt.close()
        
        avg_local = np.mean(ep_local_rewards)
        avg_ssm = np.mean(ep_ssm)
        avg_delay = np.mean(ep_delay)
        satisfy_fraction = np.mean(ep_satisfy)

        log_writer.writerow([
            ep + 1,
            ep_reward_avg,
            step_count,
            avg_local,
            np.mean(sum_rates_ep),
            np.mean(collisions_ep),
            avg_ssm,
            avg_delay,
            satisfy_fraction,
            len(buffer)
        ])
        log_file.flush()

    log_file.close()
    # -------------------------------
    # Save environment & utils code for reproducibility
    # -------------------------------

    shutil.copy("environment.py", f"{run_dir}/environment.py")
    shutil.copy("utils.py", f"{run_dir}/utils.py")
    shutil.copy("config.py", f"{run_dir}/config.py")


    plt.figure()
    plt.plot(ep_global_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Global Reward")
    plt.title("Training Convergence")
    plt.grid()
    plt.savefig(f"{run_dir}/global_reward_convergence.png")
    # plt.show()

def evaluate(env, agents, B, n_eval_episodes=10):
    eval_rewards = []

    for ep in range(n_eval_episodes):
        obs = env.reset()
        ep_reward = 0

        for t in range(cfg.episode_length):
            actions = []

            for i in range(cfg.U):
                obs_i = flatten_obs(obs[i], B)
                obs_t = T.tensor(obs_i).unsqueeze(0)
                a = agents[i].choose_action(obs_t, noise_scale=0.0)
                actions.append(a[0])

            obs, _, r_global, done, _ = env.step(actions)
            ep_reward += r_global

            if done:
                break

        eval_rewards.append(ep_reward)

    return np.mean(eval_rewards), np.std(eval_rewards)

def test():
    print("\n===== TEST MODE =====")

    env = OFDMAEnv(cfg)


    U = cfg.U
    B = cfg.B
    n_mod = len(cfg.modulations)

    # build agents
    dummy_obs = env.reset()
    state_dim = flatten_obs(dummy_obs[0], B).shape[0]

    agents = [
        Agent(cfg, agent_id=i, state_dim=state_dim, B=B, n_mod=n_mod)
        for i in range(U)
    ]

    # -------- Build Global Controller (for consistency with training) --------
    action_dim = B + B + 1 + B * n_mod

    global_controller = GlobalController(
        cfg,
        input_state_dim=state_dim,
        n_actions=action_dim,
        n_agents=cfg.U
    )

     # -------- Load BEST global critics --------
    global_controller.critic1.load_state_dict(
        T.load(
            "checkpoints/global_critic1_best.pt",
            map_location=global_controller.critic1.device
        )
    )
    global_controller.critic2.load_state_dict(
        T.load(
            "checkpoints/global_critic2_best.pt",
            map_location=global_controller.critic2.device
        )
    )

    global_controller.critic1.eval()
    global_controller.critic2.eval()

    # load BEST models
    for agent in agents:
        agent.actor.load_state_dict(
            T.load(f"checkpoints/actor_{agent.id}_best.pt",
                  map_location=agent.actor.device)
        )
        agent.actor.eval()
        agent.critic.load_state_dict(
        T.load(f"checkpoints/critic_{agent.id}_best.pt",
              map_location=agent.critic.device)
    )
    agent.critic.eval()

    mean_eval, std_eval = evaluate(env, agents, B, n_eval_episodes=cfg.test_episodes)

    print(f"[TEST] Mean Global Reward = {mean_eval:.3f}")
    print(f"[TEST] Std  Global Reward = {std_eval:.3f}")

    run_dir = f"logs/{RUN_ID}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/test_results.txt", "w") as f:
        f.write(f"Mean Reward: {mean_eval}\n")
        f.write(f"Std Reward: {std_eval}\n")

if __name__ == "__main__":
    if cfg.mode == "train":
        train()
    elif cfg.mode == "test":
        test()
    else:
        raise ValueError("cfg.mode must be 'train' or 'test'")
   