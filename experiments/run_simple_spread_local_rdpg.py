from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from rl.model.ac_network_model_rdpg import ActorNetwork, CriticNetwork
from rl.agent.model_rdpg import Trainer
import numpy as np
import torch
import time
from rl import arglist
import pickle
from rl.replay_buffer import EpisodicMemory
from copy import deepcopy
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def observation(agent, world):
    # get positions of all entities in this agent's reference frame
    entity_pos = []
    for entity in world.landmarks:  # world.entities:
        entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    # entity colors
    entity_color = []
    for entity in world.landmarks:  # world.entities:
        entity_color.append(entity.color)

    return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos)


def run(cnt):
    # load scenario from script
    scenario_name = 'simple_spread'
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # change to local observation
    scenario.observation = observation

    # create world
    world = scenario.make_world()

    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)
    env.discrete_action_input = True
    env.discrete_action_space = False

    actor = ActorNetwork(nb_agents=env.n, input_dim=10, out_dim=5)
    critic = CriticNetwork(nb_agents=env.n, input_dim=10 + 5, out_dim=1)
    memory = EpisodicMemory(limit=100000)
    agent = Trainer(actor, critic, memory)

    # initialize history
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    terminal_reward = []
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
        actions = agent.get_exploration_action(obs)
        actions = agent.process_action(actions)

        # environment step
        new_obs, rewards, done, info = env.step(actions)
        rewards = agent.process_reward(rewards)
        rewards = rewards.mean()
        episode_step += 1
        done = all(done)
        terminal = (episode_step >= arglist.max_episode_len)
        terminal = agent.process_done(done or terminal)
        # collect experience
        # obs, actions, rewards, done
        actions = agent.to_onehot(actions)
        agent.memory.append(obs, actions, rewards, terminal, training=True)

        # next observation
        obs = deepcopy(new_obs)

        # episode_rewards.append(rewards)
        rewards = rewards.item()
        for i, rew in enumerate([rewards] * env.n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        # for displaying learned policies
        if arglist.display:
            if terminal:
                time.sleep(0.1)
                env.render()
            # continue

        # for save & print history
        terminal_verbose = terminal
        if terminal:
            # save terminal state
            # process observation
            obs = agent.process_obs(obs)
            # get action & process action
            actions = agent.get_exploration_action(obs)
            actions = agent.process_action(actions)
            actions = agent.to_onehot(actions)
            # process rewards
            rewards = agent.process_reward(0.)
            rewards = rewards.mean().item()
            # process terminal
            terminal = agent.process_done(False)
            agent.memory.append(obs, actions, rewards, terminal, training=True)

            # reset environment
            obs = env.reset()
            episode_step = 0
            nb_episode += 1
            episode_rewards.append(0)
            terminal_reward.append(np.mean(rewards))

            # initialize hidden/cell states
            agent.actor.hState = None

        # increment global step counter
        train_step += 1

        # update all trainers, if not in display or benchmark mode
        loss = [np.nan, np.nan]
        if (train_step > arglist.warmup_steps) and (train_step % 600 == 0):
            # store hidden/cell state
            hState = agent.actor.hState
            # reset hidden/cell state
            agent.actor.hState = None
            # optimize actor-critic
            loss = agent.optimize()
            # recover hidden/cell state
            agent.actor.hState = hState
            loss = [loss[0].data.item(), loss[1].data.item()]
        episode_loss.append(loss)

        if verbose_step:
            if loss == [np.nan, np.nan]:
                loss = ['--', '--']
            print('step: {}, actor_loss: {}, critic_loss: {}'.format(train_step, loss[0], loss[1]))

        elif verbose_episode:
            if terminal_verbose and (len(episode_rewards) % arglist.save_rate == 0):
                print("steps: {}, episodes: {}, mean episode reward: {}, reward: {}, time: {}".format(
                    train_step, len(episode_rewards), round(np.mean(episode_rewards[-arglist.save_rate:]), 3),
                    round(np.mean(terminal_reward), 3), round(time.time() - t_start, 3)))
                terminal_reward = []
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

        # saves final episode reward for plotting training curve later
        if nb_episode > arglist.num_episodes:
            np.save('experiments/iter_{}_episode_rewards.npy'.format(cnt), episode_rewards)
            # rew_file_name = 'experiments/' + arglist.exp_name + '{}_rewards.pkl'.format(cnt)
            # with open(rew_file_name, 'wb') as fp:
            #     pickle.dump(final_ep_rewards, fp)
            # agrew_file_name = 'experiments/' + arglist.exp_name + '{}_agrewards.pkl'.format(cnt)
            # with open(agrew_file_name, 'wb') as fp:
            #     pickle.dump(final_ep_ag_rewards, fp)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break

    # np.save('history_rewards_{}.npy'.format(cnt), history_rewards)
    # np.save('history_{}.npy'.format(cnt), history)


if __name__ == '__main__':
    for cnt in range(10):
        torch.cuda.empty_cache()
        # torch.set_default_tensor_type('torch.FloatTensor')
        run(cnt)

##############################
'''
import numpy as np
from matplotlib import pyplot as plt
history_rewards = np.load('experiments/history_rewards.npy')

history_rewards.shape
plt.plot(history_rewards)
plt.show()

r = []
for i in range(len(history_rewards)//1000):
    x = history_rewards[(i * 1000):(i * 1000)+1000]
    r.append(np.mean(x))

r = np.array(r)
plt.plot(r)
plt.show()
'''