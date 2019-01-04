
# learning control
is_training = True
tau = 0.001
actor_learning_rate = 1e-2
critic_learning_rate = 1e-2
batch_size = 128
warmup_steps = 1000
update_rate = 100
max_nb_steps = 1e+6

# environment control
max_episode_len = 25
num_episodes = 40000
reward_factor = 1

# verbose control
display = False
save_rate = 1000
exp_name = 'model_'


# actions
# 0: nothing
# 1: left
# 2: right
# 3: down
# 4: up

