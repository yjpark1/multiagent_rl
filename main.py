import torch
import numpy as np
from experiments.scenarios import make_env
from rls import arglist
# proposed + model
# from rls.model.ac_network_model_multi import ActorNetwork, CriticNetwork
# from rls.agent.multiagent.model_ddpg import Trainer
# from experiments.run_fix import run

# proposed (gumbel) + model
from rls.model.ac_network_model_multi_gumbel import ActorNetwork, CriticNetwork
from rls.agent.multiagent.model_ddpg_gumbel_fix import Trainer
from experiments.run_fix import run

# BIC (gumbel)
# from rls.model.ac_network_multi_gumbel_BIC import ActorNetwork, CriticNetwork
# from rls.agent.multiagent.BIC_gumbel_fix import Trainer
# from experiments.run_fix_BIC import run

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

scenarios = ['simple_spread', 'simple_reference', 'simple_speaker_listener',
             'fullobs_collect_treasure', 'multi_speaker_listener']


for scenario_name in scenarios:
    if scenario_name == 'fullobs_collect_treasure':
        arglist.actor_learning_rate = 1e-3
        arglist.critic_learning_rate = 1e-3
    else:
        arglist.actor_learning_rate = 1e-2
        arglist.critic_learning_rate = 1e-2

    for cnt in range(10):
        # scenario_name = 'simple_spread'
        env = make_env(scenario_name, benchmark=False, discrete_action=True)
        seed = cnt + 12345678
        env.seed(seed)
        torch.cuda.empty_cache()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        dim_obs = env.observation_space[0].shape[0]
        dim_action = env.action_space[0].n

        actor = ActorNetwork(input_dim=dim_obs, out_dim=dim_action)
        critic = CriticNetwork(input_dim=dim_obs + dim_action, out_dim=1)
        run(env, actor, critic, Trainer, scenario_name, cnt=cnt)
