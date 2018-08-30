from __future__ import division

import os
import warnings

import numpy as np
import keras
import keras.backend as K
import keras.optimizers as optimizers

from rl.core import Agent
from rl.util import clone_model, clone_optimizer, get_soft_target_model_updates, huber_loss
from rl.rl_optimizer import gradients

import tensorflow as tf

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class DDPGAgent(Agent):
    """Write me
    """
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory, action_type, policy, test_policy,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, usePrioritizedMemory=False,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf, reward_factor=1.,
                 random_process=None, custom_model_objects={}, target_model_update=.001, **kwargs):
        if hasattr(actor.output, '__len__') and len(actor.output) > 1:
            raise ValueError('Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.format(actor))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError('Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))
        if critic_action_input not in critic.input:
            raise ValueError('Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError('Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.format(critic))

        super(DDPGAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
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

        self.action_type = action_type
        self.eps = 1.0
        self.reward_factor = reward_factor
        self.prioritized = usePrioritizedMemory
        self.policy = policy
        self.test_policy = test_policy

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = self.critic.input.index(critic_action_input)
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    # critic loss 함수 정의
    '''
    def critic_loss(self, y_true, y_pred):
        return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)
        # return K.mean(K.square(y_true - y_pred), axis=-1)
    '''
    def critic_loss(self, y_true, y_pred):
        loss = huber_loss(y_true, y_pred, self.delta_clip)
        # adjust updates by importance weights. Note that importance weights are just 1.0
        # (and have no effect) if not using a prioritized memory
        return K.mean(K.square(y_true - y_pred), axis=-1)

    # make loss function for Value approximation
    def critic_optimizer(self):
        # make target model
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # define critic update function
        discounted_reward = K.placeholder(shape=(None, 1))
        # actions = K.placeholder(shape=(None, self.nb_actions))
        Qvalue = self.critic.output

        loss = self.critic_loss(y_true=discounted_reward, y_pred=Qvalue)
        loss = K.mean(loss)

        # soft target updates
        updates = self.critic.optimizer.get_updates(loss=loss, params=self.critic.trainable_weights)
        grad_ops = self.critic.optimizer.get_gradients(loss=loss, params=self.critic.input[0])[0]

        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
        updates += self.critic.updates  # include other updates of the actor, e.g. for BN
        train_fn = K.function(inputs=self.critic.input + [discounted_reward, K.learning_phase()],
                              outputs=[grad_ops, loss],
                              updates=updates)
        return train_fn

    def actor_optimizer(self):
        # make target model
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        combined_inputs = []
        critic_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                critic_inputs.append(i)
        combined_inputs[self.critic_action_input_idx] = self.actor(critic_inputs)

        critic_output = self.critic(combined_inputs)

        policy = self.actor.outputs[0]
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=-1)
        loss = -K.mean(critic_output, axis=-1) + 0.05 * entropy

        updates = self.actor.optimizer.get_updates(loss=loss, params=self.actor.trainable_weights)
        '''
        # define actor update function
        # 
        grad_input = K.placeholder(shape=(None, self.nb_actions), dtype='float32')

        grads_actor = gradients(self.actor.outputs, self.actor.trainable_weights, grad_input)
        updates = self.actor.optimizer.get_updates(grads=grads_actor,
                                                   params=self.actor.trainable_weights)
        '''
        # updates = self.actor.optimizer.get_updates(grads=grad_input, params=self.actor.trainable_weights)

        # sort target updates
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN


        train_fn = K.function(inputs=self.actor.inputs + [K.learning_phase()],
                              outputs=[self.actor(critic_inputs), loss],
                              updates=updates)
        '''
        train_fn = K.function(inputs=self.actor.inputs + [grad_input, K.learning_phase()],
                              outputs=[self.actor(critic_inputs)],
                              updates=updates)
        '''
        return train_fn

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
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

        self.critic.compile(optimizer=critic_optimizer, loss='mse')
        self.actor.compile(optimizer=actor_optimizer, loss='mse')

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

    # TODO: implement pickle

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

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

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
            self.recent_action = keras.utils.to_categorical(action, num_classes=self.nb_actions)

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

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            # experiences = self.memory.sample(self.batch_size)
            if self.prioritized:
                # Calculations for current beta value based on a linear schedule.
                current_beta = self.memory.calculate_beta(self.step)
                # Sample from the memory.
                experiences = self.memory.sample(self.batch_size, current_beta)
                assert len(experiences[:-2]) == self.batch_size
            else:
                # SequentialMemory
                experiences = self.memory.sample(self.batch_size)
                assert len(experiences) == self.batch_size

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
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch) * self.reward_factor
            action_batch = np.array(action_batch)

            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            ############################################################################################################
            ############################################################################################################
            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                target_actions = np.argmax(target_actions, axis=-1)
                target_actions = keras.utils.to_categorical(target_actions, num_classes=self.nb_actions)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)

                critic_input = state0_batch_with_action + [targets] + [self.training]
                out_critic = self.critic_train_on_batch(critic_input)

                # grads = out_critic[0]
                critic_metrics = out_critic[1]

                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                if len(self.actor.inputs) >= 2:
                    inputs = state0_batch[:]
                else:
                    inputs = [state0_batch]
                # inputs += [grads]

                if self.uses_learning_phase:
                    inputs += [self.training]

                action_values = self.actor_train_on_batch(inputs)[0]
                assert action_values.shape == (self.batch_size, self.nb_actions)

            metrics = [critic_metrics]

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
