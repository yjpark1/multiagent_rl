import gym
import numpy as np
import torch
import time

from rls.model.dev.ac_network_single import ActorNetwork, CriticNetwork
from rls.agent.singleagent.ddpg import Trainer
from rls import arglist
from rls.replay_buffer import MemoryBuffer

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def run(cnt):
    # load scenario from script
    scenario_name = 'CartPole-v0'

    env = gym.make(scenario_name)
    np.random.seed(cnt+1234)
    env.seed(cnt+1234)
    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)

    actor = ActorNetwork(input_dim=4, out_dim=2)
    critic = CriticNetwork(input_dim=4 + 2, out_dim=1)
    memory = MemoryBuffer(size=1000000)
    agent = Trainer(actor, critic, memory)

    # def run():
    episode_rewards = [0.0]  # sum of rewards for all agents
    final_ep_rewards = []  # sum of rewards for training curve

    episode_loss = []
    obs = env.reset()
    episode_step = 0
    train_step = 0
    nb_episode = 0

    verbose_step = False
    verbose_episode = True
    t_start = time.time()

    print('Starting iterations...')
    while True:
        # get action
        obs = agent.process_obs(obs)
        actions, policy = agent.get_exploration_action(obs)
        actions = agent.process_action(actions)

        # environment step
        new_obs, rewards, done, info = env.step(actions)
        rewards = agent.process_reward(rewards) * arglist.reward_factor
        episode_step += 1
        terminal = False
        terminal = agent.process_done(done or terminal)
        # collect experience
        # obs, actions, rewards, new_obs, done
        # actions = agent.to_onehot(actions)
        agent.memory.add(obs, policy, rewards, agent.process_obs(new_obs), terminal)
        obs = new_obs
        # episode_rewards.append(rewards)
        rewards = rewards.item()
        episode_rewards[-1] += rewards

        # for displaying learned policies
        if arglist.display:
            if terminal:
                time.sleep(0.1)
                env.render()
            # continue

        if terminal:
            obs = env.reset()
            episode_step = 0
            nb_episode += 1
            episode_rewards.append(0)

        # increment global step counter
        train_step += 1

        # update all trainers, if not in display or benchmark mode
        loss = [np.nan, np.nan]
        if (train_step > arglist.warmup_steps) and (train_step % 1 == 0):
            loss = agent.optimize()
            loss = [loss[0].data.item(), loss[1].data.item()]

        episode_loss.append(loss)

        if verbose_step:
            if loss == [np.nan, np.nan]:
                loss = ['--', '--']
            print('step: {}, actor_loss: {}, critic_loss: {}'.format(train_step, loss[0], loss[1]))

        elif verbose_episode:
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), round((1/arglist.reward_factor) * np.mean(episode_rewards[-arglist.save_rate:]), 3),
                    round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))

        # saves final episode reward for plotting training curve later
        if nb_episode > arglist.num_episodes:
            np.save('cartpole/results/iter_{}_episode_rewards.npy'.format(cnt), episode_rewards)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break

    # np.save('history_rewards_{}.npy'.format(cnt), history_rewards)
    # np.save('history_{}.npy'.format(cnt), history)


if __name__ == '__main__':
    for cnt in range(10):
        torch.cuda.empty_cache()
        run(cnt)
