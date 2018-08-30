import os

import tensorflow as tf
import numpy as np

import keras
import keras.backend as K
from keras.layers import Input

from rl.agents.asyncAgents.core_async import AsyncAgent
from rl.util import *

from tornado import gen
from tornado.websocket import websocket_connect
from tornado.websocket import WebSocketClosedError
from tornado.websocket import StreamClosedError
from tornado.ioloop import IOLoop, PeriodicCallback


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class DiscreteA2CAgent(AsyncAgent):
    '''Actor Advantage Critic Agent for discrete actions with
    On-policy learning'''
    def __init__(self, nb_actions, actor, critic, local=True,
                 gamma=.99, beta=1e-4, reward_range=(-np.inf, np.inf),
                 delta_clip=np.inf, processor=None, custom_objects={},
                 enable_bootstrapping=True, **kwargs):
        super(DiscreteA2CAgent, self).__init__(**kwargs)

        # self.url = url
        # self.timeout = timeout
        self.nb_actions = nb_actions
        self.actor = actor
        self.critic = critic
        self.custom_objects = custom_objects

        # Parameters.
        self.local = local
        self.nb_actions = nb_actions
        self.enable_bootstrapping = enable_bootstrapping        
        self.beta = beta
        self.gamma = gamma
        self.reward_range = reward_range
        self.delta_clip = delta_clip
        self.processor = processor

        self.compiled = False
        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    # critic loss 함수 정의
    def critic_loss(self, y_true, y_pred):
        return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

    # actor loss 함수 정의
    def actor_loss(self, policy, action, advantages):
        # cross entropy loss of policy
        action_prob = K.sum(action * policy, axis=-1)
        cross_entropy = K.log(action_prob + 1e-10) * K.stop_gradient(advantages)
        cross_entropy *= -1

        # exploration entropy term
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=-1)

        return cross_entropy + 0.01 * entropy

    # make loss function for Value approximation
    def get_gradients_from_critic(self):
        discounted_reward = K.placeholder(shape=(None,))
        value = self.critic.output

        loss = self.critic_loss(y_true=discounted_reward, y_pred=value)
        loss = K.mean(loss)

        grad_ops = self.critic.optimizer.get_gradients(loss=loss, params=self.critic.trainable_weights)
        train_fn = K.function(inputs=[self.critic.input, discounted_reward],
                              outputs=[loss] + grad_ops,
                              updates=[])

        return train_fn

    def get_gradients_from_actor(self):
        action = K.placeholder(shape=(None, self.nb_actions))
        advantages = K.placeholder(shape=(None,))
        policy = self.actor.output

        loss = self.actor_loss(policy, action, advantages)
        loss = K.mean(loss)

        grad_ops = self.actor.optimizer.get_gradients(loss=loss, params=self.actor.trainable_weights)
        train_fn = K.function(inputs=[self.actor.input, action, advantages],
                              outputs=[loss] + grad_ops,
                              updates=[])
        return train_fn

    def train_critic_with_grads(self):
        # make gradient placeholder
        grad_input = [K.placeholder(shape=K.int_shape(w), dtype=K.dtype(w)) for w in self.critic.trainable_weights]
        # global model update function
        update_ops = self.critic.optimizer.get_updates(grads=grad_input,
                                                       params=self.critic.trainable_weights)
        # define keras function
        train_fn = K.function(inputs=grad_input,
                              outputs=[],
                              updates=[update_ops])

        return train_fn

    def train_actor_with_grads(self):
        # make gradient placeholder
        grad_input = [K.placeholder(shape=K.int_shape(w), dtype=K.dtype(w)) for w in self.actor.trainable_weights]
        # global model update function
        update_ops = self.actor.optimizer.get_updates(grads=grad_input,
                                                      params=self.actor.trainable_weights)
        # define keras function
        train_fn = K.function(inputs=grad_input,
                              outputs=[],
                              updates=[update_ops])

        return train_fn

    def compile(self, optimizer):
        '''
        compile에서는 loss를 정의하여 agent의 forward와 backward 함수를 생성
        actor와 critic으로 구성된 agent의 경우 actor와 critic을 개별 모델로 볼 것인지
        연결시킬 것인지 결정해야 함

        연결시킬 경우 backward flow를 따라 actor의 gradient가 critic 까지 전달
        연결시키지 않을 경우 두 모델은 개별적으로 학습됨
        '''
        metrics = []

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

        # Compile the critic. & actor
        # 'mse' is dummy loss to make prediction function
        # For training, we use <critic_optimizer>, <actor_optimizer>
        self.critic.compile(optimizer=critic_optimizer, loss='mse')
        self.actor.compile(optimizer=actor_optimizer, loss='mse')

        if self.local:
            # for local agent
            # calculate gradient
            # input: observations / output: gradient, loss
            self.critic_train_on_batch = self.get_gradients_from_critic()
            self.actor_train_on_batch = self.get_gradients_from_actor()

        else:
            # for global agent
            # update using gradient
            # input: gradient / output: loss
            self.critic_train_on_batch = self.train_critic_with_grads()
            self.actor_train_on_batch = self.train_actor_with_grads()

        self.compiled = True

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def reset_states(self):
        if getattr(self, 'local_actor', None) is not None:
            self.local_actor.reset_states()
        if getattr(self, 'local_critic', None) is not None:
            self.local_critic.reset_states()

        self.reward_accumulator = []
        self.observation_accumulator = []
        self.action_accumulator = []
        self.terminal_accumulator = []

    @property
    def metrics_names(self):
        # return self.critic.metrics_names[:] + ['mean_action']
        return ['critic_loss', 'actor_loss']

    def forward(self, observation):
        if self.processor is not None:
            observation = self.processor.process_observation(observation)
        action = self.select_action(observation)
        self.observation_accumulator.append(observation)

        action_mem = keras.utils.to_categorical(action, num_classes=self.nb_actions)
        self.action_accumulator.append(action_mem)
        return action[0]

    def select_action(self, observation):
        # Select an action.
        batch = self.process_state_batch([observation])
        action_probs = self.actor.predict_on_batch([batch])
        if self.processor is not None:
            action_probs = self.processor.process_action(action_probs)

        # stochastic actions
        # action_probs = np.squeeze(action_probs)
        actions = []
        for action_prob in action_probs:
            try:
                action = np.random.choice(self.nb_actions, 1, p=action_prob)
            except ValueError:
                action = np.random.choice(self.nb_actions, 1, p=[1. / self.nb_actions] * self.nb_actions)
            actions.append(action)
        action = np.array(actions)
        action = keras.utils.to_categorical(action, num_classes=self.nb_actions)
        action = np.argmax(action, axis=-1)
        return action

    # TODO: combine <n-step> error & <MC> error
    def discount_rewards(self, observations):
        Vs = self.critic.predict_on_batch(observations)
        if self.enable_bootstrapping:
            R = 0. if self.terminal_accumulator[-1] else Vs[-1]
        else:
            R = 0.

        # discounted rewards
        Rs = [R]
        for r, t in zip(reversed(self.reward_accumulator[:-1]), reversed(self.terminal_accumulator[:-1])):
            R = r + self.gamma * R if not t else r
            Rs.append(R)
        Rs = list(reversed(Rs))

        return Vs, Rs

    @gen.coroutine
    def backward(self, reward, terminal=False):
        metrics = [np.nan, np.nan]

        # Clip the reward to be in reward_range and perform book-keeping.
        # TODO: reward clipping using <process_reward in Process>
        self.reward_accumulator.append(reward)
        self.terminal_accumulator.append(terminal)

        perform_train = (self.training is True) and (terminal is True)
        if not perform_train:
            return metrics

        episode_steps = len(self.reward_accumulator)

        assert episode_steps == len(self.observation_accumulator)
        assert episode_steps == len(self.terminal_accumulator)
        assert episode_steps == len(self.action_accumulator)

        # Accumulate data for gradient computation.
        observations = self.process_state_batch(self.observation_accumulator)

        # TODO: make bootstrapping compatbile with LSTMs
        Vs, Rs = self.discount_rewards(observations)

        # Remove latest value, which we have no use for.
        # observations = np.array(observations[:-1])
        # actions = np.array(self.action_accumulator[:-1])
        # rewards = np.array(self.reward_accumulator[:-1])
        # terminals = np.array(self.terminal_accumulator[:-1])
        # Rs = np.array(Rs[:-1])
        # Vs = np.array(Vs[:-1])

        observations = np.array(observations)
        actions = np.array(self.action_accumulator)
        rewards = np.array(self.reward_accumulator)
        terminals = np.array(self.terminal_accumulator)
        Rs = np.array(Rs)
        Vs = np.array(Vs)

        # Rs = np.array(Rs)
        # Vs = np.array(Vs)

        Rs = Rs.reshape((episode_steps,))
        Vs = Vs.reshape((episode_steps,))
        # Ensure that everything is fine and enqueue for update.
        actions = np.squeeze(actions)

        assert observations.shape[0] == episode_steps
        assert Rs.shape == (episode_steps,)
        assert Vs.shape == (episode_steps,)
        assert rewards.shape == (episode_steps,)
        assert actions.shape == (episode_steps, self.nb_actions)
        assert terminals.shape == (episode_steps,)

        # Update critic. This also updates the local critic in the process.
        output_critic = self.critic_train_on_batch([observations, Rs])
        critic_metrics, critic_grad = output_critic[0], output_critic[1]

        advantages = Rs - Vs
        output_actor = self.actor_train_on_batch([observations, actions, advantages])
        actor_metrics, actor_grad = output_actor[0], output_actor[1]

        metrics = [critic_metrics, actor_metrics]

        # self.send_gradient_to_global_agent(self, critic_grad, actor_grad)

        # Reset state for next update round. We keep the latest data point around since we haven't
        # used it.
        self.observation_accumulator = [self.observation_accumulator[-1]]
        self.action_accumulator = [self.action_accumulator[-1]]
        self.terminal_accumulator = [self.terminal_accumulator[-1]]
        self.reward_accumulator = [self.reward_accumulator[-1]]

        return metrics

