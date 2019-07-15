import torch
import numpy as np
from experiments.scenarios import make_env
from rls import arglist
# proposed (gumbel)
# from rls.model.ac_network_multi_gumbel import ActorNetwork, CriticNetwork
# from rls.agent.multiagent.ddpg_gumbel_fix import Trainer
# from experiments.run import run, run_test

# proposed (gumbel) + model
from rls.model.ac_network_model_multi_gumbel import ActorNetwork, CriticNetwork
from rls.agent.multiagent.model_ddpg_gumbel_fix import Trainer
from experiments.run import run, run_test

# BIC (gumbel)
# from rls.model.ac_network_multi_gumbel_BIC import ActorNetwork, CriticNetwork
# from rls.agent.multiagent.BIC_gumbel_fix import Trainer
# from experiments.run_BIC import run, run_test

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

scenarios = ['simple_spread', 'simple_reference', 'simple_speaker_listener',
             'fullobs_collect_treasure', 'multi_speaker_listener']

TEST_ONLY = False
scenario_name = 'simple_spread'

for n_agent in [6, 9, 12]:
    arglist.actor_learning_rate = 1e-2
    arglist.critic_learning_rate = 1e-2

    for cnt in range(5):
        # scenario_name = 'simple_spread'
        env = make_env(scenario_name, n=n_agent, benchmark=False, discrete_action=True,
                       local_observation=True)
        seed = cnt + 12345678

        # print(env.observation_space)

        env.seed(seed)
        torch.cuda.empty_cache()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        dim_obs = env.observation_space[0].shape[0]
        if hasattr(env.action_space[0], 'high'):
            dim_action = env.action_space[0].high + 1
            dim_action = dim_action.tolist()
            action_type = 'MultiDiscrete'
        else:
            dim_action = env.action_space[0].n
            action_type = 'Discrete'

        actor = ActorNetwork(input_dim=dim_obs, out_dim=dim_action)
        critic = CriticNetwork(input_dim=dim_obs + np.sum(dim_action), out_dim=1)

        if TEST_ONLY:
            arglist.num_episodes = 100
            run_test(env, actor, critic, Trainer, scenario_name, action_type, cnt=cnt)
        else:
            scenario_name_scale = scenario_name + '_n_agent_' + str(n_agent) + '_'
            run(env, actor, critic, Trainer, scenario_name_scale, action_type, cnt=cnt)
