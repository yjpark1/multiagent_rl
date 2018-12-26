import numpy as np
import torch
import time
import pickle
from copy import deepcopy

from rls import arglist
from rls.replay_buffer import SequentialMemory

from rls.env.make_env import make_env


def run(scenario, ActorNetwork, CriticNetwork, Trainer, cnt=0):
    """function of learning agent
    """    
    torch.set_default_tensor_type('torch.FloatTensor')

    # <create world>
    world = scenario.make_world()
    world.collaborative = False

    # <create multi-agent environment>
    env = MultiAgentEnv(world, scenario.reset_world, 
                        scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = True
    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)

    # <create actor-critic networks>
    actor = ActorNetwork(input_dim=10, out_dim=5)
    critic = CriticNetwork(input_dim=10 + 5, out_dim=1)
    memory = SequentialMemory(size=1e+6)
    learner = Trainer(actor, critic, memory)

    # <initial variables>
    loss_history = []
    reward_episodes = []
    reward_episodes_by_agent = [[0. for _ in env.n]]
    episode_steps = 0
    nb_episodes = 0
    nb_steps = 0
    t_start = time.time()

    # <initialize environment>
    obs = env.reset()

    # <run interations>
    while True:
        # <get action from agent>
        obs = learner.process_obs(obs)
        actions = learner.get_exploration_action(obs)
        actions = learner.process_action(actions)

        # <run single step: send actions and return next state & reward>
        new_obs, rewards, done, info = env.step(actions)
        # append individual rewards
        for i, r in enumerate(rewards):
            reward_episodes_by_agent[-1][i] += r
        rewards = learner.process_reward(rewards)
        # append shared rewards
        reward_episodes.append(rewards)
        # update step
        nb_steps += 1

        # <get terminal condition>
        terminal = episode_steps >= arglist.max_episode_len

        # <insert single step (s, a, r, t) into replay memory>
        learner.memory.append(obs, actions, rewards, terminal, training=True)

        # <keep next state>
        obs = deepcopy(new_obs)

        if terminal:
            # <run one more step for terminate state>
            obs = learner.process_obs(obs)
            actions = learner.get_exploration_action(obs)
            actions = learner.process_action(actions)
            
            # <process rewards>
            rewards = learner.process_reward(0.)
            rewards = rewards.mean().item()

            # <process terminal>
            terminal = learner.process_done(False)

            # <insert terminal state (s, a, r, t) into replay memory>
            learner.memory.append(obs, actions, rewards, terminal, training=True)

            # <initialize environment>
            obs = env.reset()

            # <update episode count>
            nb_episodes += 1
            episode_steps = 0
            reward_episodes_by_agent.append([0. for _ in env.n])

        # <learning agent>
        do_learn = (nb_steps > arglist.warmup_steps) and (nb_steps % arglist.update_rate == 0) and arglist.is_training
        if do_learn:
            loss = learner.optimize()
            loss = np.array([x.data.item() for x in loss])
            loss_history.append(loss)

        # <update global train step count>
        nb_steps += 1

        # <verbose: print and append logs>
        if nb_steps % arglist.save_rate:
            t_end = time.time() - t_start
            msg = 'step: {}, time: {:.1f}, episodes: {}, reward_episodes: {:.3f}'.format(nb_steps,
                    t_end, nb_episodes, np.mean(reward_episodes[-arglist.save_rate:]))
            print(msg)
            learner.save_models(nb_episodes)  # save model

        # <terminate learning: steps or episodes>
        if nb_steps > arglist.nb_steps:
            print('...Finished total of {} episodes and {} steps'.format(nb_episodes, nb_steps))
            learner.save_models('fin' + str(cnt))  # save model

            print('save history...')
            d = {'loss_history': loss_history,
                 'reward_episodes': reward_episodes,
                 'reward_episodes_by_agent': reward_episodes_by_agent}
            with open('./Models/history_' + str(cnt)) as f:
                pickle.dump(d, f)

            print('done!')
            break


if __name__ == '__main__':
    import os
    from rls.model.ac_network_model_multi import ActorNetwork, CriticNetwork
    from rls.agent.singleagent.ddpg import Trainer
    from multiagent.environment import MultiAgentEnv

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    for cnt in range(10):
        torch.cuda.empty_cache()
        scenario_name = 'simple_spread'
        scenario = make_env(scenario_name, benchmark=False, discrete_action=True)

        actor = ActorNetwork(input_dim=18, out_dim=5)
        critic = CriticNetwork(input_dim=18 + 5, out_dim=1)
        run(scenario, actor, critic, Trainer, cnt=0)






