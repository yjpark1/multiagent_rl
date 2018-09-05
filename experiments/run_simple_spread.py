from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from rl.model.ac_network import ActorNetwork, CriticNetwork
from rl.agent.ddpg import Trainer
import numpy as np
import torch
import time
from rl import arglist
import pickle
from rl.replay_buffer import SequentialMemory

# load scenario from script
scenario_name = 'simple_spread'
scenario = scenarios.load(scenario_name + ".py").Scenario()

# create world
world = scenario.make_world()

# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
print('observation shape: ', env.observation_space)
print('action shape: ', env.action_space)
env.discrete_action_input = True
env.discrete_action_space = False

actor = ActorNetwork(input_dim=18, out_dim=5)
critic = CriticNetwork(input_dim=18 + 5, out_dim=1)
memory = SequentialMemory(limit=1000)
agent = Trainer(actor, critic, memory)

# def run():
episode_rewards = [0.0]  # sum of rewards for all agents
obs = env.reset()
episode_step = 0
train_step = 0

print('Starting iterations...')
while True:
    # get action
    obs = agent.process_obs(obs)
    actions = agent.get_exploration_action(obs)
    actions = agent.process_action(actions)
    agent.memory.append(obs, actions, np.float(0.), False)

    # environment step
    new_obs, rewards, done, info = env.step(actions)
    rewards = agent.process_reward(rewards)
    rewards = np.mean(rewards)
    episode_step += 1
    done = all(done)
    terminal = (episode_step >= arglist.max_episode_len)

    # collect experience
    # obs, actions, rewards, new_obs, done
    agent.memory.append(new_obs, actions, rewards, done or terminal)
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
    loss = ['--', '--']
    if train_step > arglist.warmup_steps:
        loss = agent.optimize()

    print('step: {}, actor_loss: {}, critic_loss: {}'.format(train_step, loss[0], loss[1]))
    # saves final episode reward for plotting training curve later
    if len(episode_rewards) > arglist.num_episodes:
        print('...Finished total of {} episodes.'.format(len(episode_rewards)))
        break

'''
if __name__ == '__main__':
    run()
'''
