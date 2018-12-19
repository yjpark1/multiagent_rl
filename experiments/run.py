import gym
import numpy as np
import torch
import time

import multiagent.scenarios as scenarios

from rls import arglist
from rls.replay_buffer import SequentialMemory


def run(scenario, ActorNetwork, CriticNetowrk, Trainer, 
        cnt=0, local_obs=True):
    """function of learning agent
    """    
    torch.set_default_tensor_type('torch.FloatTensor')

    # <create world>
    world = scenario.make_world()

    # <create multi-agent environment>
    env = MultiAgentEnv(world, scenario.reset_world, 
                        scenario.reward, scenario.observation)
    env.discrete_action_input = True
    env.discrete_action_space = False
    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)

    # <create actor-critic networks>
    actor = ActorNetwork(input_dim=10, out_dim=5)
    critic = CriticNetwork(input_dim=10 + 5, out_dim=1)
    memory = MemoryBuffer(size=1e+6)
    learner = Trainer(actor, critic, memory)

    # <initial variables>
    reward_episodes = []
    reward_episodes_by_agent = []
    step_episode = 0
    step_train = 0
    nb_episode = 0
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
        rewards = agent.process_reward(rewards)
        episode_step += 1

        # <insert single step (s, a, r, t) into replay memory>
        learner.memory.append(obs, actions, rewards, terminal, training=True)

        # <keep next state>
        obs = deepcopy(new_obs)

        # <get terminal condition>
        terminal = terminal or done

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
            nb_episode += 1

        # <learning agent>
        do_learn = (train_step > arglist.warmup_steps) and (train_step % 100 == 0) and arglist.is_training
        if do_learn:
            loss = learner.optimize()
            loss = np.array([x.data.item() for x in loss])
            episode_loss.append(loss)

        # <update global train step count>
        step_train += 1

        # <verbose: print and append logs>

        # <terminate learning: steps or episodes>
        if step_train > arglist.nb_steps:
            print('...Finished total of {} episodes and {} steps'.format(nb_episode, step_train)
            break



if __name__ == '__main__':
    import os
    from rls.model.ac_network_single import ActorNetwork, CriticNetwork
    from rls.agent.singleagent.ddpg import Trainer
    from multiagent.environment import MultiAgentEnv

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    run(scenario, ActorNetwork, CriticNetowrk, Trainer, 
        cnt=0, local_obs=True)

    for cnt in range(10):
        torch.cuda.empty_cache()        
        run(cnt)






