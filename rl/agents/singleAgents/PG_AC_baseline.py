from __future__ import division
from collections import deque
import os
import warnings
import keras
import numpy as np
import keras.backend as K
import keras.optimizers as optimizers
from keras.utils.np_utils import to_categorical
from keras.layers import Input

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
                 train_interval=1, memory_interval=1, delta_clip=np.inf,
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

    # critic loss 함수 정의
    def critic_loss(self, y_true, y_pred):
        return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

    # actor loss 함수 정의
    def actor_loss(self, action, advantages):
        def policy_gradient_loss(y_true, y_pred):
            # policy의 cross entropy loss
            action_prob = K.sum(action * y_pred, axis=-1)
            cross_entropy = K.log(action_prob + 1e-10) * advantages
            # cross_entropy = -K.mean(cross_entropy)
        
            # 탐색을 지속적으로 하기 위한 엔트로피 오류
            entropy = K.sum(y_pred * K.log(y_pred + 1e-10), axis=-1)
            # entropy = K.mean(entropy)
        
            # 두 오류함수를 더해 최종 오류함수를 만듬
            # train_on_batch를 사용하기 때문에 K.mean 함수는 제거함
            return cross_entropy + 0.001 * entropy
        return policy_gradient_loss

    def compile(self, optimizer, metrics=[]):
        ''' 
        compile에서는 loss를 정의하여 agent의 forward와 backward 함수를 생성
        actor와 critic으로 구성된 agent의 경우 actor와 critic을 개별 모델로 볼 것인지
        연결시킬 것인지 결정해야 함

        연결시킬 경우 backward flow를 따라 actor의 gradient가 critic 까지 전달
        연결시키지 않을 경우 두 모델은 개별적으로 학습됨
        '''
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

        # Compile target networks. 
        # target network의 경우 forward pass 만 이용하기 때문에 
        # optimizer 와 loss는 상관없음
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')
        
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        
        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(target=self.target_critic,
                                                           source=self.critic,
                                                           tau=self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer,
                            loss=self.critic_loss,
                            metrics=critic_metrics)

        # Compile the actor.
        # loss 계산을 위해 action, advantages를 actor 모델에 포함
        # one-hot action vector from replays
        action = Input(shape=(self.nb_agents, self.nb_actions))
        advantages = Input(shape=(1,)) # Q - V
        self.actor = keras.models.Model(inputs=[self.actor.inputs[0], action, advantages], outputs=self.actor.outputs[0])
        
        if self.target_model_update < 1.:
            # Include soft target model updates.
            actor_updates = get_soft_target_model_updates(target=self.target_actor,
                                                          source=self.actor,
                                                          tau=self.target_model_update)
            actor_optimizer = AdditionalUpdatesOptimizer(actor_optimizer, actor_updates)
        
        self.actor.compile(optimizer=actor_optimizer,
                           loss=self.actor_loss(self.actor.inputs[1], self.actor.inputs[2]))
        
        self.compiled = True

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
            self.target_actor.reset_states()

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
                action = np.random.choice(self.nb_actions, 1, p=[1./self.nb_actions] * self.nb_actions)

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

            action_batch = np.squeeze(action_batch)
            assert action_batch.shape == (self.batch_size, self.nb_agents, self.nb_actions)

            # critic update
            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_V = self.target_critic.predict_on_batch(state1_batch)
                assert target_V.shape == (self.batch_size, 1)
                
                discounted_reward_batch = self.gamma * target_V
                terminal1_batch = terminal1_batch.reshape((self.batch_size, 1))
                discounted_reward_batch *= terminal1_batch
                
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
                V = self.critic.predict_on_batch(x=state0_batch)
                advantages = targets - V
                loss_actor = self.actor.train_on_batch(x=[state0_batch, action_batch, advantages],
                                                       y=np.zeros((self.batch_size, self.nb_agents, self.nb_actions)))

                # logger.info('loss_actor: ' + str(loss_actor))

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics
