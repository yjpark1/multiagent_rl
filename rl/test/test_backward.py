import numpy as np

from rl.agents.multiAgents.multi_ddpg import MA_DDPGAgent
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint

import keras
from keras.optimizers import Adam

from model.env_starcarft import StarCraftEnvironment
from model import GlobalVariable as gvar

from rl.MultiAgentPolicy import starcraft_multiagent_eGreedyPolicy
from rl.policy import LinearAnnealedPolicy
from model.ac_networks import actor_net, critic_net
from rl.callbacks import TrainHistoryLogCallback

import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = gvar.cuda_device

# define environment
agent_name = 'starcraft_minigame_vulture_zealot'
env_details = {'ally': ['verture']*2, 'enemy': ['zealot']*3, 'state_dim': (64, 64, 3 + 2),
               'dim_state_2D': (64, 64, 5), 'dim_state_1D': (2, )}
env = StarCraftEnvironment(agent_name, env_details)

# define policy
policy_minigame = starcraft_multiagent_eGreedyPolicy(nb_agents=env.nb_agents,
                                                     nb_actions=env.nb_actions)
policy = LinearAnnealedPolicy(policy_minigame, attr='eps', value_max=1.,
                              value_min=.1, value_test=.05, nb_steps=400000)
test_policy = starcraft_multiagent_eGreedyPolicy(nb_agents=env.nb_agents,
                                                 nb_actions=env.nb_actions,
                                                 eps=0.05)

keras.backend.clear_session()
actor = actor_net(env)
critic = critic_net(env)

# build actor/critic network
model_path = 'save_model/{}_weights.h5f'.format(agent_name)
memory = SequentialMemory(limit=50000, window_length=1)
histcallback = TrainHistoryLogCallback(file_path='save_model/', plot_interval=100)

# define policy
policy_minigame = starcraft_multiagent_eGreedyPolicy(nb_agents=env.nb_agents,
                                                     nb_actions=env.nb_actions)
policy = LinearAnnealedPolicy(policy_minigame, attr='eps', value_max=1.,
                              value_min=.1, value_test=.05, nb_steps=100000)
test_policy = starcraft_multiagent_eGreedyPolicy(nb_agents=env.nb_agents,
                                                 nb_actions=env.nb_actions,
                                                 eps=0.05)

agent = MA_DDPGAgent(nb_agents=env.nb_agents, nb_actions=env.nb_actions,
                     actor=actor, critic=critic, action_type='discrete',
                     critic_action_input=critic.inputs[2:4], train_interval=10,
                     batch_size=32, memory=memory, nb_steps_warmup_critic=5000,
                     nb_steps_warmup_actor=5000, policy=policy, test_policy=test_policy,
                     gamma=.99, target_model_update=1e-3)

agent.compile([Adam(lr=1e-3), Adam(lr=1e-3)], metrics=['mae'])

actor.summary()
critic.summary()

#################################
#################################
#################################
agent.training = 1
agent.step = 6000
# Train the network on a single stochastic batch.
can_train_either = (agent.step > agent.nb_steps_warmup_critic or agent.step > agent.nb_steps_warmup_actor)
if can_train_either and agent.step % agent.train_interval == 0:
    # experiences = agent.memory.sample(agent.batch_size)
    mem = np.load('memory.npy')
    mem = mem.tolist()
    experiences = mem.sample(agent.batch_size)
    # assert len(experiences) == agent.batch_size

    # Start by extracting the necessary parameters (we use a vectorized implementation).
    state0_batch = []
    reward_batch = []
    action_batch = []
    terminal1_batch = []
    state1_batch = []

    for e in experiences:
        state0_batch.append(e.state0)
        state1_batch.append(e.state1)
        reward_batch.append(e.reward)
        action_batch.append(e.action)
        terminal1_batch.append(0. if e.terminal1 else 1.)

    # Prepare and validate parameters.
    state0_batch = agent.process_state_batch(state0_batch)
    state1_batch = agent.process_state_batch(state1_batch)
    terminal1_batch = np.array(terminal1_batch)
    reward_batch = np.array(reward_batch) * agent.reward_factor
    action_batch = agent.process_action_batch(action_batch)

    action_batch_xy, action_batch_type = action_batch[0], action_batch[1]

    assert reward_batch.shape == (agent.batch_size,)
    assert terminal1_batch.shape == reward_batch.shape
    # assert action_batch.shape == (agent.batch_size, agent.nb_agents, agent.nb_actions)
    #####################################################################################
    #####################################################################################
    # Update critic, if warm up is over.
    if agent.step > agent.nb_steps_warmup_critic:
        # make target action using target_actor
        target_actions = agent.target_actor.predict_on_batch(state1_batch)

        target_actions_xy = target_actions[0]
        target_actions_type = target_actions[1]

        target_actions_xy = np.argmax(target_actions_xy, axis=-1)
        target_actions_type = np.argmax(target_actions_type, axis=-1)

        target_actions_xy = keras.utils.to_categorical(target_actions_xy, num_classes=agent.nb_actions[0])
        assert target_actions_xy.shape == (agent.batch_size, agent.nb_agents, agent.nb_actions[0])

        target_actions_type = keras.utils.to_categorical(target_actions_type, num_classes=agent.nb_actions[1])
        assert target_actions_type.shape == (agent.batch_size, agent.nb_agents, agent.nb_actions[1])

        # state1 + target_action
        state1_batch_with_action = [state1_batch[0], state1_batch[1], target_actions_xy, target_actions_type]
        target_q_values = agent.target_critic.predict_on_batch(state1_batch_with_action).flatten()
        assert target_q_values.shape == (agent.batch_size, )

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = agent.gamma * target_q_values
        discounted_reward_batch *= terminal1_batch
        assert discounted_reward_batch.shape == reward_batch.shape
        targets = (reward_batch + discounted_reward_batch).reshape(agent.batch_size, 1)

        # state0 + action_batch
        state0_batch_with_action = [state0_batch[0], state0_batch[1], action_batch_xy, action_batch_type]
        critic_input = state0_batch_with_action + [targets] + [agent.training]
        out_critic = agent.critic_train_on_batch(critic_input)

        critic_metrics = out_critic[0]

        if agent.processor is not None:
            metrics += agent.processor.metrics

    # Update actor, if warm up is over.
    if agent.step > agent.nb_steps_warmup_actor:
        # TODO: implement metrics for actor
        inputs = state0_batch[:]

        if agent.uses_learning_phase:
            inputs += [agent.training]

        agent.actor_train_on_batch(inputs)

    metrics = [critic_metrics]