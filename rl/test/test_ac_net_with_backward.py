import numpy as np

from rl.agents.multiAgents.multi_rdpg import MA_RDPGAgent
from rl.memory import SequentialMemory
from rl.memory import EpisodicMemory
from rl.callbacks import ModelIntervalCheckpoint

import tensorflow as tf
import keras
import keras.backend as K
from keras.optimizers import Adam
from rl.rl_optimizer import Adam_rl

from model.env_starcarft import StarCraftEnvironment
from model.ac_networks import actor_net, critic_net
from model import GlobalVariable as gvar

from rl.MultiAgentPolicy import starcraft_multiagent_eGreedyPolicy
from rl.policy import LinearAnnealedPolicy
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

batch_size = 3

with gvar.graph.as_default():
    critic = critic_net(env, model_type='train_model', batch_size=batch_size)
    actor = actor_net(env, model_type='train_model', batch_size=batch_size)
    policy_actor = actor_net(env, model_type='policy_model', batch_size=1)
    policy_critic = critic_net(env, model_type='policy_model', batch_size=1)


critic.compile('Adam', 'mse')
critic._make_predict_function()
critic._make_train_function()

actor.compile('Adam', 'categorical_crossentropy')
actor._make_predict_function()
actor._make_train_function()

step = 10
actor_x = [np.ones(shape=(3, 2, step, 64, 64, 5)), np.ones(shape=(3, 2, step, 2))]
actor_y = [np.ones(shape=(3, 2, step, 4096)), np.ones(shape=(3, 2, step, 3))]
critic_x = actor_x + actor_y
critic_y = [np.ones(shape=(3, step, 1))]


c = critic.fit(critic_x, critic_y, batch_size=3)
print(c)
a = actor.fit(actor_x, actor_y, batch_size=3)
print(a)

for _ in range(20):
    a = actor.train_on_batch(actor_x, actor_y)
    print(a)
    c = critic.train_on_batch(critic_x, critic_y)
    print(c)


# build actor/critic network
model_path = 'save_model/{}_weights.h5f'.format(agent_name)
# memory = SequentialMemory(limit=50000, window_length=1)
memory = EpisodicMemory(limit=50000, window_length=1)
histcallback = TrainHistoryLogCallback(file_path='save_model/', plot_interval=100)
print('step1...')
agent = MA_RDPGAgent(nb_agents=env.nb_agents, nb_actions=env.nb_actions, memory=memory,
                     actor=actor, critic=critic, policy_actor=policy_actor, policy_critic=policy_critic,
                     critic_action_input=critic.inputs[2:4], action_type='discrete',
                     policy=policy, test_policy=test_policy, env_details=env_details,
                     batch_size=batch_size, nb_max_steps_recurrent_unrolling=40,
                     nb_steps_warmup_critic=500, nb_steps_warmup_actor=500, implementation=1,
                     train_interval=10)

agent.compile([Adam(lr=1e-3, clipnorm=40.), Adam(lr=1e-3, clipnorm=40.)], metrics=['mae'])
actor.summary()
critic.summary()
time.sleep(1)

mem = np.load('episodic_memory.npy')
mem = mem.tolist()

terminal = True
#####################
if terminal:
    agent.policy_actor.reset_states()
    agent.policy_critic.reset_states()

agent.step = 2000
agent.training = True
metrics = [np.nan for _ in agent.metrics_names]
print('compile done...')
# Train the network on a single stochastic batch.
can_train_either = (agent.step > agent.nb_steps_warmup_critic or agent.step > agent.nb_steps_warmup_actor)
if can_train_either and agent.step % agent.train_interval == 0:
    experiences = mem.sample(agent.batch_size)
    assert len(experiences) == agent.batch_size

    lengths = [len(seq) for seq in experiences]
    maxlen = np.max(lengths)
    print(maxlen)

    # Start by extracting the necessary parameters (we use a vectorized implementation).
    state0_batch = [[] for _ in range(len(experiences))]
    reward_batch = [[] for _ in range(len(experiences))]
    action_batch = [[] for _ in range(len(experiences))]
    terminal1_batch = [[] for _ in range(len(experiences))]
    state1_batch = [[] for _ in range(len(experiences))]

    for sequence_idx, sequence in enumerate(experiences):
        for e in sequence:
            state0_batch[sequence_idx].append(e.state0)
            state1_batch[sequence_idx].append(e.state1)
            reward_batch[sequence_idx].append(e.reward)
            action_batch[sequence_idx].append(e.action)
            terminal1_batch[sequence_idx].append(0. if e.terminal1 else 1.)

        # Apply padding.
        while len(state0_batch[sequence_idx]) < maxlen:
            state0_batch[sequence_idx].append([np.zeros(shape=(agent.nb_agents, ) + agent.env_details['dim_state_2D']),
                                               np.zeros(shape=(agent.nb_agents, ) + agent.env_details['dim_state_1D'])])
            state1_batch[sequence_idx].append([np.zeros(shape=(agent.nb_agents, ) + agent.env_details['dim_state_2D']),
                                               np.zeros(shape=(agent.nb_agents, ) + agent.env_details['dim_state_1D'])])
            reward_batch[sequence_idx].append(0.)
            action_batch[sequence_idx].append([np.zeros(shape=(agent.nb_agents, agent.nb_actions[0])),
                                               np.zeros(shape=(agent.nb_agents, agent.nb_actions[1]))])
            terminal1_batch[sequence_idx].append(1.)

    state0_batch = agent.process_state_batch(state0_batch)
    state1_batch = agent.process_state_batch(state1_batch)
    terminal1_batch = np.array(terminal1_batch)
    reward_batch = np.array(reward_batch) * agent.reward_factor
    action_batch = agent.process_action_batch(action_batch)

    action_batch_xy, action_batch_type = action_batch[0], action_batch[1]

    assert reward_batch.shape == (agent.batch_size, maxlen)
    assert terminal1_batch.shape == reward_batch.shape
    # Update critic, if warm up is over.
    if agent.step > agent.nb_steps_warmup_critic and terminal:
        if agent.is_recurrent:
            # Reset states before training on the entire sequence.
            agent.critic.reset_states()
            agent.target_critic.reset_states()

        # make target action using target_actor
        target_actions = agent.target_actor.predict_on_batch(state1_batch)

        target_actions_xy = target_actions[0]
        target_actions_type = target_actions[1]

        target_actions_xy = np.argmax(target_actions_xy, axis=-1)
        target_actions_type = np.argmax(target_actions_type, axis=-1)

        target_actions_xy = keras.utils.to_categorical(target_actions_xy, num_classes=agent.nb_actions[0])
        assert target_actions_xy.shape == (agent.batch_size, agent.nb_agents, maxlen, agent.nb_actions[0])

        target_actions_type = keras.utils.to_categorical(target_actions_type, num_classes=agent.nb_actions[1])
        assert target_actions_type.shape == (agent.batch_size, agent.nb_agents, maxlen, agent.nb_actions[1])

        # state1 + target_action
        state1_batch_with_action = [state1_batch[0], state1_batch[1], target_actions_xy, target_actions_type]
        target_q_values = agent.target_critic.predict_on_batch(state1_batch_with_action)
        target_q_values = np.squeeze(target_q_values, -1)
        assert target_q_values.shape == (agent.batch_size, maxlen)

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = agent.gamma * target_q_values
        discounted_reward_batch *= terminal1_batch
        assert discounted_reward_batch.shape == reward_batch.shape
        targets = (reward_batch + discounted_reward_batch).reshape(agent.batch_size, maxlen, 1)

        # ##########################
        # In the recurrent case, we support splitting the sequences into multiple
        # chunks. Each chunk is then used as a training example. The reason for this is that,
        # for too long episodes, the unrolling in time during backpropagation can exceed the
        # memory of the GPU (or, to a lesser degree, the RAM if training on CPU).
        # state0 + action_batch
        state0_batch_with_action = [state0_batch[0], state0_batch[1], action_batch_xy, action_batch_type]
        print('forward done...')
        if agent.is_recurrent and agent.nb_max_steps_recurrent_unrolling:
            assert targets.ndim == 3
            steps = targets.shape[1]  # (batch_size, steps, 1)
            nb_chunks = int(np.ceil(float(steps) / float(agent.nb_max_steps_recurrent_unrolling)))

            chunks = []
            for chunk_idx in range(nb_chunks):
                start = chunk_idx * agent.nb_max_steps_recurrent_unrolling
                end = start + agent.nb_max_steps_recurrent_unrolling

                sa = [x[:, :, start:end, ...] for x in state0_batch_with_action]
                t = targets[:, start:start + agent.nb_max_steps_recurrent_unrolling, ...]

                chunks.append(sa + [t] + [agent.training])
        else:
            critic_input = state0_batch_with_action + [targets] + [agent.training]
            chunks = [critic_input]

        critic_metrics = []
        start = time.time()
        for chunk_train in chunks:
            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ms = agent.critic_train_on_batch(chunk_train)[0]
            print('.', end='')
            critic_metrics.append(ms)
        print('critic done: ' + str(time.time() - start))
        critic_metrics = np.mean(critic_metrics)
        metrics[0] = critic_metrics
        ##########################
        # state0 + action_batch
        # state0_batch_with_action = [state0_batch[0], state0_batch[1], action_batch_xy, action_batch_type]
        # critic_input = state0_batch_with_action + [targets] + [agent.training]
        # out_critic = agent.critic_train_on_batch(critic_input)

    # Update actor, if warm up is over.
    if agent.step > agent.nb_steps_warmup_actor and terminal:
        if agent.is_recurrent:
            # Reset states before training on the entire sequence.
            agent.actor.reset_states()
            agent.target_actor.reset_states()

        # TODO: implement metrics for actor
        inputs = state0_batch[:]

        if agent.is_recurrent and agent.nb_max_steps_recurrent_unrolling:
            assert targets.ndim == 3
            steps = targets.shape[1]  # (batch_size, steps, 1)
            nb_chunks = int(np.ceil(float(steps) / float(agent.nb_max_steps_recurrent_unrolling)))

            chunks = []
            for chunk_idx in range(nb_chunks):
                start = chunk_idx * agent.nb_max_steps_recurrent_unrolling
                end = start + agent.nb_max_steps_recurrent_unrolling

                s = [x[:, :, start:end, ...] for x in inputs]
                if agent.uses_learning_phase:
                    s += [agent.training]

                chunks.append(s + [agent.training])
        else:
            actor_input = inputs + [agent.training]
            chunks = [actor_input]

        start = time.time()
        for chunk_train in chunks:
            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            a = agent.actor_train_on_batch(chunk_train)[0]
            print('.', end='')
        print('actor done: ' + str(time.time() - start))

