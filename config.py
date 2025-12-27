import numpy as np
import math
class Config:
    
    # -----------------------
    # General
    # -----------------------
    mode = "train"
    episodes = 500
    episode_length = 100
    gamma = 0.99   # Discount factor
    tau = 0.005    # Soft update parameter
    seed = 100
    save_interval = 10
    test_episodes = 20
    use_deterministic = False



    # -----------------------
    # environment params
    # -----------------------
    cell_radius_km = 0.5
    bs_position = (0.0, 0.0)
    shadowing_std_db = 6.0   # standard deviation of log-normal shadowing in dB
    P_max = 1.0
    N0_dBm_per_Hz = -174     # Noise PSD
    W = 180e3
    N0_dBm = N0_dBm_per_Hz + 10*np.log10(W)
    N0 = 10 ** ((N0_dBm - 30) / 10)   # Noise per RB (W) 
    CR_min = 0.1
    CR_max = 0.9
    num_images_min = 5
    num_images_max = 20
    service_levels = [1, 2, 3]
    max_sla = max(service_levels)
    image_size = 3000 # bit
    delay_th_min = 0.05   # 100 ms
    delay_th_max = 0.5    # 600 ms
    min_gap = 0.2



    # -----------------------
    # reward weighting
    # -----------------------
    theta1 = 0.4
    theta2 = 0.3
    theta3 = 0.3
    lambda1 = 1.0
    lambda2 = 1.0
    lambda3 = 1.0
    lambda4 = 1.0
    lambda_power = 1.0   # tune later
    theta_satisfy = 0.1  # Bonus for satisfying SSM_min
    delay_cap = 2.0
    eps_cr = 0.1

    

    # -----------------------
    # Multi-agent environment
    # -----------------------
    U = 2               # number of agents
    B = 2               # number of RBs
    modulations = [4, 16, 64]   # example list


    # -----------------------
    # JCM Feature-based tables
    # -----------------------
    total_features = 128       
    min_features = 76          
    feature_step = 2           
    jcm_table_dir = "jcm_tables"   
    jcm_table_prefix = "table_symbols_"  
    jcm_D_index = 0     # indices inside each JCM cell (0-based)
    jcm_PQ_index = 4    # indices inside each JCM cell (0-based)
    jcm_snr_column = "snr" 



    # -----------------------
    # Learning
    # -----------------------
    lambda_loss = 1.0         # Lambda for balancing local and global loss in actor update
    replay_buffer_size = 100000
    batch_size = 64
    actor_lr = 1e-4
    critic_lr = 1e-3
    gaussian_noise_sigma = 0.3

    policy_delay = 2   # Policy update delay factor

    # -----------------------
    # Actor/critic layer definitions
    # -----------------------
    actor_layers = [1024, 512]                 # local actor network (per agent)
    local_critic_layers = [1024, 512, 256]     # local critic network (per agent)
    global_critic_layers = [1024, 512, 256]    # global critic (3 hidden layers)


cfg = Config()
