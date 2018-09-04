from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from rl.model.ac_network import ActorNetwork, CriticNetwork
from rl.agent.ddpg import Trainer
import numpy as np
import torch
import time
from rl import arglist
import pickle
from rl.replay_buffer import ReplayBuffer

# load scenario from script
scenario_name = 'simple_reference'
scenario = scenarios.load(scenario_name + ".py").Scenario()

# create world
world = scenario.make_world()

# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
actor = ActorNetwork(input_dim=21, out_dim=2)
critic = CriticNetwork(input_dim=21, out_dim=1)
memory = ReplayBuffer()
agent = Trainer(actor, critic, memory)

env.observation_space
env.action_space

def run():
    episode_rewards = [0.0]  # sum of rewards for all agents
    obs = env.reset()
    episode_step = 0
    train_step = 0

    ###############################
    print('Starting iterations...')
    while True:
        # get action
        obs = np.array(obs)
        obs = np.expand_dims(obs, axis=0)
        obs = torch.tensor(obs, dtype=torch.float)

        actions = agent.action(obs)
        # environment step
        new_obs, rewards, done, info = env.step(actions)
        episode_step += 1
        done = all(done)
        terminal = (episode_step >= arglist.max_episode_len)
        # collect experience
        agent.experience(obs, actions, rewards, new_obs, done, terminal)
        obs = new_obs
        episode_rewards[-1] += rewards

        if done or terminal:
            obs = env.reset()
            episode_step = 0
            episode_rewards.append(0)

        # increment global step counter
        train_step += 1

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # update all trainers, if not in display or benchmark mode
        loss = None
        agent.preupdate()
        loss = agent.update()

        # saves final episode reward for plotting training curve later
        if len(episode_rewards) > arglist.num_episodes:
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break
