'''Multi-Agent DDPG (new)
<TODO>
1. multi agent policy
2. bidirectional lstm multi agent
3. lstm policy
'''
from __future__ import division

import os
import warnings

import numpy as np
import keras
import keras.backend as K
import keras.optimizers as optimizers
from keras.layers import TimeDistributed, Reshape

from rl.core import Agent
from rl.util import (clone_model, clone_optimizer,
                     get_soft_target_model_updates, huber_loss,
                     get_object_config)

from model import GlobalVariable as gvar
from rl.rl_optimizer import gradients

import tensorflow as tf
import time

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class MA_RDPGAgent(Agent):

    """Write me
    """
    def __init__(self, nb_agents, nb_actions, memory,
                 actor, critic, policy_actor, policy_critic,
                 critic_action_input, action_type, env_details,
                 policy, test_policy, is_recurrent=True,
                 gamma=.99, batch_size=32, nb_max_steps_recurrent_unrolling=20,
                 nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=5, memory_interval=1,
                 delta_clip=np.inf, reward_factor=1., implementation=1,
                 random_process=None, custom_model_objects={},
                 target_model_update=.001, **kwargs):


        if hasattr(actor.output, '__len__') and len(actor.output) != 2:
            raise ValueError(
                'Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.format(actor))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError(
                'Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))
        if (critic_action_input[0] not in critic.input) or (critic_action_input[1] not in critic.input):
            raise ValueError(
                'Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError(
                'Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.format(
                    critic))

        super(MA_RDPGAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        # Parameters.
        self.nb_agents = nb_agents
        self.nb_actions = nb_actions
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects

        # Added parameters.
        self.action_type = action_type
        self.reward_factor = reward_factor
        self.policy = policy
        self.test_policy = test_policy
        self.env_details = env_details

        # recurrent policy
        self.is_recurrent = is_recurrent
        self.policy_actor = policy_actor
        self.policy_critic = policy_critic
        self.nb_max_steps_recurrent_unrolling = nb_max_steps_recurrent_unrolling

        # implementation version
        # 1: use mean Q value to update actor
        # 2: use critic gradient to update actor
        self.implementation = implementation

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = [self.critic.input.index(x) for x in self.critic_action_input]
        self.memory = memory

        if self.is_recurrent:
            # memory = kwargs['memory']
            if not memory.is_episodic:
                raise ValueError('Recurrent RL requires an episodic memory. You are trying to use it with memory={} instead.'.format(memory))

        # State.
        self.compiled = False
        self.reset_states()


    def get_config(self):
        config = super(MA_RDPGAgent, self).get_config()
        config['nb_max_steps_recurrent_unrolling'] = self.nb_max_steps_recurrent_unrolling
        config['policy'] = get_object_config(self.policy)
        return config


    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase


    # critic loss 함수 정의
    def critic_loss(self, y_true, y_pred):
        loss = huber_loss(y_true, y_pred, self.delta_clip)
        return K.mean(loss, axis=[1, 2])


    # make loss function for Value approximation
    def critic_optimizer(self):
        # make target model
        self.target_critic = keras.models.clone_model(self.critic)
        self.target_critic.compile(optimizer='sgd', loss='mse')
        
        # define critic update function
        discounted_reward = K.placeholder(shape=(self.batch_size, None, 1))
        Qvalue = self.critic.outputs[0]

        loss = self.critic_loss(y_true=discounted_reward, y_pred=Qvalue)
        loss = K.mean(loss)

        # soft target updates
        updates = []
        updates += self.critic.optimizer.get_updates(loss=loss, params=self.critic.trainable_weights)

        if self.implementation == 2:
            # gradient of action input layer
            # TODO: multiple action grdient
            self.grad_ops = self.critic.optimizer.get_gradients(loss=loss, params=self.critic.input[2:4])

        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)

        if self.policy_critic is not self.critic:
            # Update the policy model after every training step.
            updates += get_soft_target_model_updates(self.policy_critic, self.critic, 1.)

        # updates += self.critic.updates  # include other updates of the actor, e.g. for BN
        
        if self.implementation == 1:
            train_fn = K.function(inputs=self.critic.inputs + [discounted_reward, K.learning_phase()],
                                  outputs=[loss],  # [grad_ops, loss]
                                  updates=updates)
        elif self.implementation == 2:
            train_fn = K.function(inputs=self.critic.inputs + [discounted_reward, K.learning_phase()],
                                  outputs=[loss],
                                  updates=updates)

        return train_fn


    def loss_control(self, target_xy, policy_xy, policy_type):
        # get enemy position
        # target_xy = self.actor.inputs[0][:, :, :, :, :, 1]
        target_xy = TimeDistributed(TimeDistributed(Reshape((4096,))))(target_xy)
        ce = K.sum(target_xy * K.log(policy_xy + 1e-10), axis=-1)

        is_attack = K.argmax(policy_type, axis=-1) == 2

        return is_attack * ce


    def actor_optimizer(self):
        # make target model
        self.target_actor = keras.models.clone_model(self.actor)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        
        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        combined_inputs = []
        critic_inputs = []
        for input_tensor in self.critic.inputs:
            if input_tensor in self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(input_tensor)
                critic_inputs.append(input_tensor)

        combined_inputs[self.critic_action_input_idx[0]:self.critic_action_input_idx[1] + 1] = self.actor(critic_inputs)
        critic_output = self.critic(combined_inputs)

        # entropy for exploration
        # policy_0 = self.actor.outputs[0]
        # policy_1 = self.actor.outputs[1]
        '''
        policy_0 = combined_inputs[2]
        policy_1 = combined_inputs[3]
        
        entropy_0 = K.sum(policy_0 * K.log(policy_0 + 1e-10), axis=3)
        entropy_1 = K.sum(policy_1 * K.log(policy_1 + 1e-10), axis=3)

        entropy_0 = K.mean(entropy_0, axis=(1, 2))
        entropy_1 = K.mean(entropy_1, axis=(1, 2))

        target_xy = combined_inputs[0][:, :, :, :, :, 1]
        loss_attack = self.loss_control(target_xy=target_xy, policy_xy=policy_0, policy_type=policy_1)
        loss_attack = K.mean(loss_attack, axis=(1, 2))
        '''
        # implementation version 1: update actor by Q-value
        if self.implementation == 1:
            loss = -K.mean(critic_output, axis=[1, 2])
            # loss += 0.01 * (entropy_0 + entropy_1)
            # loss += 0.01 * loss_attack
            loss = K.mean(loss)

            updates = []
            updates += self.actor.optimizer.get_updates(loss=loss, params=self.actor.trainable_weights)
            
            # sort target updates
            if self.target_model_update < 1.:
                # Include soft target model updates.
                updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)

            if self.policy_actor is not self.actor:
                # Update the policy model after every training step.
                updates += get_soft_target_model_updates(self.policy_actor, self.actor, 1.)

            # updates += self.actor.updates  # include other updates of the actor, e.g. for BN
            
            train_fn = K.function(inputs=critic_inputs + [K.learning_phase()],
                                  outputs=[loss],
                                  updates=updates)

        # implementation version 2: update actor by critic gradient
        elif self.implementation == 2:
            updates = []
            # define actor update function by gradient
            # grad_input = K.placeholder(shape=(self.batch_size, self.nb_actions), dtype='float32')
            grads_actor = gradients(self.actor.outputs, self.actor.trainable_weights, self.grad_ops)
            updates += self.actor.optimizer.get_updates(grads=grads_actor,
                                                        params=self.actor.trainable_weights)

            # TODO: multiple action grdient
            train_fn = K.function(inputs=critic_inputs + [K.learning_phase()],
                                  outputs=[],
                                  updates=updates)

        return train_fn


    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError(
                    'More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        with gvar.graph.as_default():
            self.critic.compile(optimizer=critic_optimizer, loss='mse')
            self.critic._make_predict_function()
            self.critic._make_train_function()

            self.actor.compile(optimizer=actor_optimizer, loss='mse')
            self.actor._make_predict_function()
            self.actor._make_train_function()

            self.policy_critic.compile(optimizer='sgd', loss='mse')
            self.policy_actor.compile(optimizer='sgd', loss='mse')

            self.critic_train_on_batch = self.critic_optimizer()
            self.actor_train_on_batch = self.actor_optimizer()

        self.compiled = True


    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()


    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)


    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())


    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()
            self.policy_actor.reset_states()
            self.policy_critic.reset_states()


    def process_state_batch(self, batch):
        b2d = []
        b1d = []
        for sample in batch:
            s2d = np.concatenate([np.expand_dims(x[0], 1) for x in sample], axis=1)
            s1d = np.concatenate([np.expand_dims(x[1], 1) for x in sample], axis=1)
            b2d.append(s2d)
            b1d.append(s1d)
        batch = [np.stack(b2d, axis=0), np.stack(b1d, axis=0)]

        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)


    def process_action_batch(self, batch):
        b2d = []
        b1d = []
        for sample in batch:
            s2d = np.concatenate([np.expand_dims(x[0], 1) for x in sample], axis=1)
            s1d = np.concatenate([np.expand_dims(x[1], 1) for x in sample], axis=1)
            b2d.append(s2d)
            b1d.append(s1d)
        batch = [np.stack(b2d, axis=0), np.stack(b1d, axis=0)]

        return batch


    def select_action(self, state):
        batch = self.process_state_batch([state])

        if self.is_recurrent:
            action = self.policy_actor.predict_on_batch(batch)

        else:
            action = self.actor.predict_on_batch(batch)

        assert len(action) == 2

        if self.training:
            action = self.policy.select_action(preference=action)
        else:
            action = self.test_policy.select_action(preference=action)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action


    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy

        if self.action_type == 'discrete':
            # Book-keeping: action
            # one-hot action
            self.recent_action = [keras.utils.to_categorical(action[0], self.nb_actions[0]),
                                  keras.utils.to_categorical(action[1], self.nb_actions[1])]

        elif self.action_type == 'continuous':
            self.recent_action = action

        # Book-keeping: observation
        self.recent_observation = observation

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        if terminal:
            self.policy_actor.reset_states()
            self.policy_critic.reset_states()

        # Train the network on a single stochastic batch.
        can_train_either = (self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor)
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            lengths = [len(seq) for seq in experiences]
            maxlen = np.max(lengths)

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
                    state0_batch[sequence_idx].append([np.zeros(shape=(self.nb_agents, ) + self.env_details['dim_state_2D']),
                                                       np.zeros(shape=(self.nb_agents, ) + self.env_details['dim_state_1D'])])
                    state1_batch[sequence_idx].append([np.zeros(shape=(self.nb_agents, ) + self.env_details['dim_state_2D']),
                                                       np.zeros(shape=(self.nb_agents, ) + self.env_details['dim_state_1D'])])
                    reward_batch[sequence_idx].append(0.)
                    action_batch[sequence_idx].append([np.zeros(shape=(self.nb_agents, self.nb_actions[0])),
                                                       np.zeros(shape=(self.nb_agents, self.nb_actions[1]))])
                    terminal1_batch[sequence_idx].append(1.)

            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch) * self.reward_factor
            action_batch = self.process_action_batch(action_batch)

            action_batch_xy, action_batch_type = action_batch[0], action_batch[1]

            assert reward_batch.shape == (self.batch_size, maxlen)
            assert terminal1_batch.shape == reward_batch.shape

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                # Reset states before training on the entire sequence.
                self.critic.reset_states()
                self.target_critic.reset_states()

                # make target action using target_actor
                target_actions = self.target_actor.predict_on_batch(state1_batch)

                target_actions_xy = target_actions[0]
                target_actions_type = target_actions[1]

                target_actions_xy = np.argmax(target_actions_xy, axis=-1)
                target_actions_type = np.argmax(target_actions_type, axis=-1)

                target_actions_xy = keras.utils.to_categorical(target_actions_xy, num_classes=self.nb_actions[0])
                assert target_actions_xy.shape == (self.batch_size, self.nb_agents, maxlen, self.nb_actions[0])

                target_actions_type = keras.utils.to_categorical(target_actions_type, num_classes=self.nb_actions[1])
                assert target_actions_type.shape == (self.batch_size, self.nb_agents, maxlen, self.nb_actions[1])

                # state1 + target_action
                state1_batch_with_action = [state1_batch[0], state1_batch[1], target_actions_xy, target_actions_type]
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action)
                target_q_values = np.squeeze(target_q_values, -1)
                assert target_q_values.shape == (self.batch_size, maxlen)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, maxlen, 1)

                # ##########################
                # In the recurrent case, we support splitting the sequences into multiple
                # chunks. Each chunk is then used as a training example. The reason for this is that,
                # for too long episodes, the unrolling in time during backpropagation can exceed the
                # memory of the GPU (or, to a lesser degree, the RAM if training on CPU).
                # state0 + action_batch
                state0_batch_with_action = [state0_batch[0], state0_batch[1], action_batch_xy, action_batch_type]

                if self.is_recurrent and self.nb_max_steps_recurrent_unrolling:
                    assert targets.ndim == 3
                    steps = targets.shape[1]  # (batch_size, steps, 1)
                    nb_chunks = int(np.ceil(float(steps) / float(self.nb_max_steps_recurrent_unrolling)))

                    chunks = []
                    for chunk_idx in range(nb_chunks):
                        start = chunk_idx * self.nb_max_steps_recurrent_unrolling
                        end = start + self.nb_max_steps_recurrent_unrolling

                        sa = [x[:, :, start:end, ...] for x in state0_batch_with_action]
                        t = targets[:, start:start + self.nb_max_steps_recurrent_unrolling, ...]

                        chunks.append(sa + [t] + [self.training])
                else:
                    critic_input = state0_batch_with_action + [targets] + [self.training]
                    chunks = [critic_input]

                critic_metrics = []
                for chunk_train in chunks:
                    # Finally, perform a single update on the entire batch. We use a dummy target since
                    # the actual loss is computed in a Lambda layer that needs more complex input. However,
                    # it is still useful to know the actual target to compute metrics properly.
                    ms = self.critic_train_on_batch(chunk_train)[0]
                    print('critic train...', end='')
                    critic_metrics.append(ms)

                critic_metrics = np.mean(critic_metrics)
                metrics[0] = critic_metrics
                ##########################
                # state0 + action_batch
                # state0_batch_with_action = [state0_batch[0], state0_batch[1], action_batch_xy, action_batch_type]
                # critic_input = state0_batch_with_action + [targets] + [self.training]
                # out_critic = self.critic_train_on_batch(critic_input)

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # Reset states before training on the entire sequence.
                self.actor.reset_states()
                self.target_actor.reset_states()

                # TODO: implement metrics for actor
                inputs = state0_batch[:]

                if self.is_recurrent and self.nb_max_steps_recurrent_unrolling:
                    assert targets.ndim == 3
                    steps = targets.shape[1]  # (batch_size, steps, 1)
                    nb_chunks = int(np.ceil(float(steps) / float(self.nb_max_steps_recurrent_unrolling)))

                    chunks = []
                    for chunk_idx in range(nb_chunks):
                        start = chunk_idx * self.nb_max_steps_recurrent_unrolling
                        end = start + self.nb_max_steps_recurrent_unrolling

                        s = [x[:, :, start:end, ...] for x in inputs]
                        if self.uses_learning_phase:
                            s += [self.training]

                        chunks.append(s + [self.training])
                else:
                    actor_input = inputs + [self.training]
                    chunks = [actor_input]

                for chunk_train in chunks:
                    # Finally, perform a single update on the entire batch. We use a dummy target since
                    # the actual loss is computed in a Lambda layer that needs more complex input. However,
                    # it is still useful to know the actual target to compute metrics properly.
                    self.actor_train_on_batch(chunk_train)
                    print('actor train...', end='')

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)
