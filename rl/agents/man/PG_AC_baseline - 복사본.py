from __future__ import division
from collections import deque
import os
import warnings
import keras
import numpy as np
import keras.backend as K
import keras.optimizers as optimizers
from keras.utils.np_utils import to_categorical

from rl.core import Agent
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import *
from rl.MultiAgentPolicy import MA_EpsGreedyQPolicy
from util.CustomLog import cLogger
logger = cLogger.getLogger()

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class PGAC_baseline(Agent):
    """Write me
    """
    def __init__(self, nb_agents, nb_actions, actor, critic, memory,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={},
                 target_model_update=.001, **kwargs):
        if hasattr(actor.output, '__len__') and len(actor.output) > 1:
            raise ValueError(
                'Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.format(actor))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError(
                'Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))

        super(PGAC_baseline, self).__init__(**kwargs)

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
            warnings.warn(
                '`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(
                    delta_range[1]))
            delta_clip = delta_range[1]

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

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

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

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(target=self.target_critic, source=self.critic, tau=self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Compile the actor.
        self.actor_train_fn = self.actor_optimizer()

        self.compiled = True

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.nb_agents, self.nb_actions])
        advantages = K.placeholder(shape=[None, 1])

        policy = self.actor.output

        # 정책 크로스 엔트로피 오류함수
        action_prob = K.sum(action * policy, axis=-1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.mean(cross_entropy)

        # 탐색을 지속적으로 하기 위한 엔트로피 오류
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=-1)
        entropy = K.mean(entropy)

        # 탐색을 지속적으로 하기 위한 <agent 엔트로피 >
        '''
        ind = K.argmax(policy, axis=-1)
        one_hot = K.one_hot(ind, self.nb_actions)
        prob = K.sum(one_hot, axis=1) / self.nb_agents
        entropy_agent = K.sum(prob * K.log(prob + K.epsilon()), axis=-1)
        entropy_agent = K.mean(entropy_agent)
        '''
        # 두 오류함수를 더해 최종 오류함수를 만듬
        # loss = cross_entropy + 0. * entropy + 0. * entropy_agent
        loss = cross_entropy + 0.001 * entropy

        optimizer = self.actor.optimizer
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        # updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)

        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        # cross_entropy, entropy, entropy_agent
        return train

    def load_weights(self, filepath):
        dirname, filepath = os.path.split(filepath)
        filename, extension = os.path.splitext(filepath)
        actor_filepath = dirname + '/' + filename + '_actor' + extension
        critic_filepath = dirname + '/' + filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        dirname, filepath = os.path.split(filepath)
        filename, extension = os.path.splitext(filepath)
        actor_filepath = dirname + '/' + filename + '_actor' + extension
        critic_filepath = dirname + '/' + filename + '_critic' + extension
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
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def predict_action_prob(self, state):
        batch = self.process_state_batch(state)
        action = self.actor.predict_on_batch(batch)
        assert action.shape == (1, self.nb_agents, self.nb_actions)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample
            assert noise.shape == action.shape
            action += noise
        return action

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action_probs = self.predict_action_prob(state)  # TODO: move this into policy
        if self.processor is not None:
            action_probs = self.processor.process_action(action_probs)

        # stochastic actions
        action_probs = np.squeeze(action_probs)
        actions = []
        for action_prob in action_probs:
            try:
                action = np.random.choice(self.nb_actions, 1, p=action_prob)
            except ValueError:
                action = np.random.choice(self.nb_actions, 1, p=[0.2] * self.nb_actions)

            actions.append(action)
        action = np.array(actions)
        action = keras.utils.to_categorical(action, num_classes=self.nb_actions)
        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

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
        # memory 저장
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
            # memory 꺼내오기 및 전처리
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0[0])
                state1_batch.append(e.state1[0])
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape

            # action_batch = np.squeeze(action_batch, axis=1)
            action_batch = np.squeeze(action_batch)
            # action_batch = to_categorical(action_batch, self.nb_actions)
            # print('action_batch shape: ', action_batch.shape)
            assert action_batch.shape == (self.batch_size, self.nb_agents, self.nb_actions)

            # critic update
            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                target_actions = to_categorical(target_actions.argmax(axis=-1), self.nb_actions)
                # assert target_actions.shape == (self.batch_size, self.nb_agents, self.nb_actions)

                target_q_values = self.target_critic.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, 1)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values

                # terminal1_batch = np.expand_dims(terminal1_batch, axis=-1)  #
                # terminal1_batch = np.repeat(terminal1_batch, repeats=self.nb_agents, axis=-1)  #
                # terminal1_batch = np.expand_dims(terminal1_batch, axis=-1)  #
                terminal1_batch = terminal1_batch.reshape((self.batch_size, 1))
                discounted_reward_batch *= terminal1_batch

                # print("reward_batch shape: ", reward_batch.shape)
                # reward_batch = np.expand_dims(reward_batch, axis=-1)  #
                # reward_batch = np.repeat(reward_batch, repeats=self.nb_agents, axis=-1)  #
                # reward_batch = np.expand_dims(reward_batch, axis=-1)  #
                reward_batch = reward_batch.reshape((self.batch_size, 1))
                assert discounted_reward_batch.shape == reward_batch.shape

                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                metrics = self.critic.train_on_batch(x=state0_batch, y=targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                state0_batch = state0_batch.astype('float32')
                values = self.critic.predict_on_batch(x=state0_batch)                
                advantages = targets - values
                #if terminal:
                loss_actor = self.actor_train_fn([state0_batch, action_batch, advantages])
                # loss_actor = self.actor_train_fn([state0_batch, target_actions, advantages])

                logger.info('loss_actor: ' + str(loss_actor))
                # assert action_values.shape == (self.batch_size, self.nb_agents, self.nb_actions)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics
