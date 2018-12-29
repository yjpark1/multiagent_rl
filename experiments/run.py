import numpy as np
import torch
import time
import pickle
from copy import deepcopy

from rls import arglist
from rls.replay_buffer import SequentialMemory
from rls.model.ac_network_model_multi import ActorNetwork, CriticNetwork
from rls.agent.multiagent.model_ddpg import Trainer
from experiments.scenarios import make_env


def run(env, actor, critic, Trainer, cnt=0):
    """function of learning agent
    """    
    torch.set_default_tensor_type('torch.FloatTensor')
    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)

    # <create actor-critic networks>
    memory = SequentialMemory(limit=int(1e+6))
    learner = Trainer(actor, critic, memory)

    # <initial variables>
    loss_history = []
    reward_episodes = [0.]
    reward_episodes_by_agent = [[0. for _ in range(env.n)]]
    nb_episodes = 1
    nb_steps = 1
    episode_steps = 0
    t_start = time.time()

    # <initialize environment>
    obs = env.reset()

    # <run interations>
    while True:
        # <get action from agent>
        obs = learner.process_obs(obs)
        actions = learner.get_exploration_action(obs)

        # <run single step: send actions and return next state & reward>
        new_obs, rewards, done, info = env.step(actions)
        # append individual rewards
        for i, r in enumerate(rewards):
            reward_episodes_by_agent[-1][i] += r
        rewards = learner.process_reward(rewards)
        # append shared rewards
        reward_episodes[-1] += rewards.item()
        # update step
        nb_steps += 1
        episode_steps +=1

        # <get terminal condition>
        terminal = episode_steps >= arglist.max_episode_len

        # <insert single step (s, a, r, t) into replay memory>
        actions = learner.process_action(actions)
        learner.memory.append(obs, actions, rewards, terminal, training=True)

        # <keep next state>
        obs = deepcopy(new_obs)

        if terminal:
            # <run one more step for terminate state>
            obs = learner.process_obs(obs)
            actions = learner.get_exploration_action(obs)
            
            # <process rewards>
            reward_episodes_by_agent.append([0. for _ in range(env.n)])
            rewards = learner.process_reward(0.)
            reward_episodes.append(rewards.item())

            # <process terminal>
            terminal = learner.process_done(False)

            # <insert terminal state (s, a, r, t) into replay memory>
            actions = learner.process_action(actions)
            learner.memory.append(obs, actions, rewards, terminal, training=True)

            # <initialize environment>
            obs = env.reset()

            # <update episode count>
            nb_episodes += 1
            episode_steps = 0

            # <verbose: print and append logs>
            if nb_episodes % arglist.save_rate == 0:
                t_end = time.time() - t_start
                msg = 'step: {}, time: {:.1f}, episodes: {}, reward_episodes: {:.3f}'.format(nb_steps,
                                                                                             t_end, nb_episodes,
                                                                                             np.mean(reward_episodes[
                                                                                                     -arglist.save_rate:]))
                print(msg)
                t_start = time.time()

        # <learning agent>
        do_learn = (nb_steps > arglist.warmup_steps) and (nb_steps % arglist.update_rate == 0) and arglist.is_training
        if do_learn:
            loss = learner.optimize()
            loss = np.array([x.data.item() for x in loss])
            loss_history.append(loss)

        # <terminate learning: steps or episodes>
        if nb_steps > arglist.max_nb_steps:
            print('...Finished total of {} episodes and {} steps'.format(nb_episodes, nb_steps))
            learner.save_models('fin' + str(cnt))  # save model

            print('save history...')
            d = {'loss_history': loss_history,
                 'reward_episodes': reward_episodes,
                 'reward_episodes_by_agent': reward_episodes_by_agent}
            with open('Models/history_' + str(cnt) + '.pkl', 'wb') as f:
                pickle.dump(d, f)

            print('done!')
            break


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    for cnt in range(10):
        scenario_name = 'simple_spread'
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
        run(env, actor, critic, Trainer, cnt=cnt)






