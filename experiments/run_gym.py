import numpy as np
import torch
import time
import pickle
from copy import deepcopy

from rls import arglist
from rls.replay_buffer import ReplayBuffer



def run(env, actor, critic, Trainer, scenario_name=None,
        model=False, model_advance=False, flag_train=True,
        action_type='Discrete', cnt=0):
    """
    function of learning agent
    """
    if flag_train:
        exploration_mode = 'train'
    else:
        exploration_mode = 'test'

    torch.set_default_tensor_type('torch.FloatTensor')
    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)

    # <create actor-critic networks>
    memory = ReplayBuffer(size=1e+6)
    learner = Trainer(actor, critic, memory,
                      model=model, model_advance=model_advance, action_type=action_type)

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()
    best_prev = -np.inf

    print('Starting iterations...')
    while True:
        # get action
        if action_type == 'Discrete':
            action_n = learner.get_exploration_action(obs_n, mode=exploration_mode)
            action_n_env = np.argmax(action_n)

        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n_env)
        # make shared reward
        rew_shared = np.sum(rew_n) * 0.1

        episode_step += 1
        done = np.all(done_n)
        terminal = (episode_step >= arglist.max_episode_len) or done
        # collect experience
        learner.memory.add(obs_n, action_n, rew_shared, new_obs_n, float(done))
        obs_n = new_obs_n

        for i, rew in enumerate([rew_n]):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        # increment global step counter
        train_step += 1

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.001)
            env.render()
            continue

        # update all trainers, if not in display or benchmark mode
        # <learning agent>
        do_learn = (train_step > arglist.warmup_steps) and (
                train_step % arglist.update_rate == 0) and arglist.is_training
        if flag_train and do_learn:
            loss = learner.optimize()

        if flag_train:
            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                # print statement depends on whether or not there are adversaries
                avg_rewards = np.mean(episode_rewards[-arglist.save_rate:])
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), avg_rewards,
                    round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                # model check point
                if best_prev >= avg_rewards:
                    is_best = True
                    best_prev = avg_rewards
                else:
                    is_best = False
                learner.save_training_checkpoint(fname=scenario_name + '_fin_' + str(cnt), is_best=is_best)

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes or train_step > arglist.max_train_step:
                hist = {'reward_episodes': episode_rewards, 'reward_episodes_by_agents': agent_rewards}
                file_name = 'Models/history_' + scenario_name + '_' + str(cnt) + '.pkl'
                with open(file_name, 'wb') as fp:
                    pickle.dump(hist, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                learner.save_models(scenario_name + '_bestchk_' + str(cnt))  # save model
                break
        else:
            # save model, display testing output
            if terminal and (len(episode_rewards) % 10 == 0):
                # print statement depends on whether or not there are adversaries
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-10:]),
                    round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-10:]))
                final_ep_rewards.append(np.mean(episode_rewards[-10:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-10:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                hist = {'reward_episodes_own': episode_rewards,
                        'reward_episodes_adv': episode_rewards,
                        'reward_episodes_by_agents': agent_rewards}
                file_name = 'Models/test_history_' + scenario_name + '_' + str(cnt) + '.pkl'
                with open(file_name, 'wb') as fp:
                    pickle.dump(hist, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


if __name__ == '__main__':
    from rls.model.ac_network_single import ActorNetwork, CriticNetwork
    from rls.agent.singleagent.model_ddpg import Trainer
    import gym
    import os

    arglist.actor_learning_rate = 1e-3
    arglist.critic_learning_rate = 1e-3

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    cnt = 13
    ENV_NAME = 'CartPole-v0'
    env = gym.make(ENV_NAME)
    env.n = 1
    seed = cnt + 12345678
    env.seed(seed)
    # np.random.seed(seed)
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dim_obs = env.observation_space.shape[0]
    dim_action = env.action_space.n
    action_type = 'Discrete'

    flag_AML = True
    actor = ActorNetwork(input_dim=dim_obs, out_dim=dim_action, model=flag_AML)
    critic = CriticNetwork(input_dim=dim_obs + np.sum(dim_action), out_dim=1, model=flag_AML)
    run(env, actor, critic, Trainer, ENV_NAME,
        model=flag_AML, model_advance=True, flag_train=True,
        action_type=action_type, cnt=cnt)



