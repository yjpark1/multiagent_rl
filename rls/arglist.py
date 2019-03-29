# network parameter
hidden = 128

# environment control
max_episode_len = 1000 # 25
num_episodes = 10000 # 40000
max_train_step = 500000
reward_factor = 1

# learning control
is_training = True
tau = 1e-2
actor_learning_rate = None
critic_learning_rate = None
batch_size = 32
warmup_steps = batch_size * 10
warmup_steps_rnd = 50000
update_rate = 1 # 100
max_nb_steps = 1e+6
gamma = 0.99 # 0.95

# verbose control
display = False
save_rate = 100
exp_name = 'model_'

# train model path appendix
appx = 'env_partial/proposed+gumbel/'

# actions
# 0: nothing
# 1: left
# 2: right
# 3: down
# 4: up

