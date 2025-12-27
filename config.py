class Config:
    # General
    episodes = 500
    episode_length = 100
    gamma = 0.99   # Discount factor
    tau = 0.005    # Soft update parameter

     # environment params
    shadowing_std_db = 6.0   # standard deviation of log-normal shadowing in dB
    P_max = 1.0
    N0 = 1e-9
    W = 180e3
    CR_min = 0.1
    CR_max = 0.9
    num_images_min = 3
    num_images_max = 12
    service_levels = [1, 2, 3]
    image_size = 128 # bit



    # reward weighting
    theta1 = 0.4
    theta2 = 0.3
    theta3 = 0.3
    lambda1 = 1.0
    lambda2 = 1.0
    lambda3 = 1.0
    lambda4 = 1.0
    


    # Multi-agent environment
    U = 4               # number of agents
    B = 8               # number of RBs
    modulations = [4, 16, 64]   # example list

    # JCM lookup table
    jcm_class_csv = "jcm_sample_classification.csv"   # <-- NEW (required by environment.py)
    jcm_rec_csv = "jcm_sample_reconstruction.csv"


    # Learning
    replay_buffer_size = 100000
    batch_size = 64
    actor_lr = 1e-4
    critic_lr = 1e-3
    gaussian_noise_sigma = 0.3

    policy_delay = 2   # Policy update delay factor

    # -----------------------
    # NEW: Actor/critic layer definitions
    # -----------------------
    # local actor network (per agent)
    actor_layers = [1024, 512]

    # local critic network (per agent)
    local_critic_layers = [1024, 512, 256]

    # global critic (3 hidden layers)
    global_critic_layers = [1024, 512, 256]

cfg = Config()
