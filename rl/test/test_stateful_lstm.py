import numpy as np

from rl.agents.multiAgents.multi_rdpg import MA_RDPGAgent
from rl.memory import SequentialMemory
from rl.memory import EpisodicMemory
from rl.callbacks import ModelIntervalCheckpoint

import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import (Input, Conv2D, MaxPooling2D,
                          Concatenate, Bidirectional, LSTM,
                          Dense, TimeDistributed, Flatten,
                          ConvLSTM2D, Conv2DTranspose, GlobalAveragePooling2D,
                          Reshape)
from keras.models import Model

from model.env_starcarft import StarCraftEnvironment
from model import GlobalVariable as gvar

from rl.MultiAgentPolicy import starcraft_multiagent_eGreedyPolicy
from rl.policy import LinearAnnealedPolicy

from rl.callbacks import TrainHistoryLogCallback

import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
# build actor/critic network
#################################actor######################################
# observation inputs
stateful = True
batchsize = 3
state_2D = Input(batch_shape=(batchsize, env.nb_agents, None) + env.state_dim)
state_1D = Input(batch_shape=(batchsize, env.nb_agents, None) + (2, ))
# <actor network>
# 2d-state encoder
h_2d = TimeDistributed(TimeDistributed(Conv2D(32, 4, activation='relu', padding='same', strides=4,
                                                kernel_regularizer=l2(1e-3))))(state_2D)
h_2d = TimeDistributed(TimeDistributed(Conv2D(64, 4, activation='relu', padding='same', strides=4,
                                                kernel_regularizer=l2(1e-3))))(h_2d)
h_2d = TimeDistributed(TimeDistributed(GlobalAveragePooling2D()))(h_2d)
# combine 1d state & 2d state
h = Concatenate()([h_2d, state_1D])
# <multi-agent with RNN>
# rnn along time steps: need <return sequences = True>
h = TimeDistributed(LSTM(64, activation='relu', return_sequences=True,
                            stateful=True, kernel_regularizer=l2(1e-3)))(h)
# h = TimeDistributed(TimeDistributed(keras.layers.Activation('relu')))(h)
# LSTM policy network
# rnn along agents
h = keras.layers.Permute((2, 1, 3))(h)
h = TimeDistributed(Bidirectional(LSTM(32, activation='relu', return_sequences=True)))(h)
h = keras.layers.Permute((2, 1, 3))(h)
# h = TimeDistributed(TimeDistributed(keras.layers.Activation('relu')))(h)
# action_xy
h_xy = TimeDistributed(TimeDistributed(keras.layers.Reshape(target_shape=(8, 8, 1))))(h)
h_xy = TimeDistributed(TimeDistributed(Conv2DTranspose(1, 4, activation='relu', padding='same',
                                                        strides=(2, 2),
                                                        kernel_regularizer=l2(1e-3))))(h_xy)
h_xy = TimeDistributed(TimeDistributed(Conv2DTranspose(1, 4, activation='linear', padding='same',
                                                        strides=(4, 4),
                                                        kernel_regularizer=l2(1e-3))))(h_xy)
h_xy = TimeDistributed(TimeDistributed(Flatten()))(h_xy)
# h_xy = TimeDistributed(TimeDistributed(Reshape((4096,))))(h_xy)
action_xy = TimeDistributed(TimeDistributed(keras.layers.Activation('softmax')))(h_xy)
# action_type
action_type = Dense(3, activation='softmax',
                    kernel_regularizer=l2(1e-3), name='action_type')(h)
actor = Model(inputs=[state_2D, state_1D],
              outputs=[action_xy, action_type])

################################policy actor################################
# observation inputs
stateful = True
batchsize = 1
state_2D = Input(batch_shape=(batchsize, env.nb_agents, None) + env.state_dim)
state_1D = Input(batch_shape=(batchsize, env.nb_agents, None) + (2, ))
# <actor network>
# 2d-state encoder
h_2d = TimeDistributed(TimeDistributed(Conv2D(32, 4, activation='relu', padding='same', strides=4,
                                                kernel_regularizer=l2(1e-3))))(state_2D)
h_2d = TimeDistributed(TimeDistributed(Conv2D(64, 4, activation='relu', padding='same', strides=4,
                                                kernel_regularizer=l2(1e-3))))(h_2d)
h_2d = TimeDistributed(TimeDistributed(GlobalAveragePooling2D()))(h_2d)
# combine 1d state & 2d state
h = Concatenate()([h_2d, state_1D])
# <multi-agent with RNN>
# rnn along time steps
h = TimeDistributed(LSTM(64, activation='relu', return_sequences=True,
                            stateful=True, kernel_regularizer=l2(1e-3)))(h)
h = keras.layers.Lambda(lambda x: x[:, :, -1, :])(h)
# LSTM policy network
# rnn along agents
h = Bidirectional(LSTM(32, activation='relu', return_sequences=True))(h)
# action_xy
h_xy = TimeDistributed(keras.layers.Reshape(target_shape=(8, 8, 1)))(h)
h_xy = TimeDistributed(Conv2DTranspose(1, 4, activation='relu', padding='same',
                                        strides=(2, 2),
                                        kernel_regularizer=l2(1e-3)))(h_xy)
h_xy = TimeDistributed(Conv2DTranspose(1, 4, activation='linear', padding='same',
                                        strides=(4, 4),
                                        kernel_regularizer=l2(1e-3)))(h_xy)
h_xy = TimeDistributed(Flatten())(h_xy)
action_xy = TimeDistributed(keras.layers.Activation('softmax'))(h_xy)
# action_type
action_type = Dense(3, activation='softmax',
                    kernel_regularizer=l2(1e-3), name='action_type')(h)
policy_actor = Model(inputs=[state_2D, state_1D],
                     outputs=[action_xy, action_type])
#################################critic#####################################
# observation inputs
batchsize = 3
stateful = True
state_2D = Input(batch_shape=(batchsize, env.nb_agents, None) + env.state_dim)
state_1D = Input(batch_shape=(batchsize, env.nb_agents, None) + (2,))
# action input
input_action_xy = Input(batch_shape=(batchsize, env.nb_agents, None, 64 * 64,))
input_action_type = Input(batch_shape=(batchsize, env.nb_agents, None, 3,))
h_action_xy = TimeDistributed(TimeDistributed(keras.layers.Reshape((64, 64, 1))))(input_action_xy)
h_2d = Concatenate()([state_2D, h_action_xy])
# 2d-state encoder
h_2d = TimeDistributed(TimeDistributed(Conv2D(32, 4, activation='relu', padding='same', strides=4,
                                                kernel_regularizer=l2(1e-3))))(h_2d)
h_2d = TimeDistributed(TimeDistributed(Conv2D(64, 4, activation='relu', padding='same', strides=4,
                                                kernel_regularizer=l2(1e-3))))(h_2d)
h_2d = TimeDistributed(TimeDistributed(GlobalAveragePooling2D()))(h_2d)
# combine 1d state & 2d state
h = Concatenate()([h_2d, state_1D, input_action_type])
# <multi-agent with RNN>
# rnn along time steps: need <return sequences = True>
h = TimeDistributed(LSTM(64, activation='relu', return_sequences=True, stateful=stateful,
                         kernel_regularizer=l2(1e-3)))(h)
# LSTM policy network
# rnn along agents and return all outputs of the time steps
h = keras.layers.Permute((2, 1, 3))(h)
h = TimeDistributed(Bidirectional(LSTM(32, activation='relu', return_sequences=False)))(h)
# Q value
q_value = Dense(1, activation='linear', kernel_regularizer=l2(1e-3))(h)
critic = Model(inputs=[state_2D, state_1D, input_action_xy, input_action_type],
               outputs=[q_value])

################################policy critic###############################
# observation inputs
stateful = True
batchsize = 1
state_2D = Input(batch_shape=(batchsize, env.nb_agents, None) + env.state_dim)
state_1D = Input(batch_shape=(batchsize, env.nb_agents, None) + (2, ))
# action input
input_action_xy = Input(batch_shape=(batchsize, env.nb_agents, None, 64 * 64,))
input_action_type = Input(batch_shape=(batchsize, env.nb_agents, None, 3,))
h_action_xy = TimeDistributed(TimeDistributed(keras.layers.Reshape((64, 64, 1))))(input_action_xy)
h_2d = Concatenate()([state_2D, h_action_xy])
# 2d-state encoder
h_2d = TimeDistributed(TimeDistributed(Conv2D(32, 4, activation='relu', padding='same', strides=4,
                                                kernel_regularizer=l2(1e-3))))(h_2d)
h_2d = TimeDistributed(TimeDistributed(Conv2D(64, 4, activation='relu', padding='same', strides=4,
                                                kernel_regularizer=l2(1e-3))))(h_2d)
h_2d = TimeDistributed(TimeDistributed(GlobalAveragePooling2D()))(h_2d)
# combine 1d state & 2d state
h = Concatenate()([h_2d, state_1D, input_action_type])
# <multi-agent with RNN>
# rnn along time steps
h = TimeDistributed(LSTM(64, activation='relu', return_sequences=True, stateful=stateful,
                            kernel_regularizer=l2(1e-3)))(h)
h = keras.layers.Lambda(lambda x: x[:, :, -1, :])(h)
# LSTM policy network
# rnn along agents
h = Bidirectional(LSTM(32, activation='relu', return_sequences=False))(h)
# Q value
q_value = Dense(1, activation='linear', kernel_regularizer=l2(1e-3))(h)
policy_critic = Model(inputs=[state_2D, state_1D, input_action_xy, input_action_type],
                      outputs=[q_value])
############################################################################

memory = EpisodicMemory(limit=50000, window_length=1)
print('step1...')
agent = MA_RDPGAgent(nb_agents=env.nb_agents, nb_actions=env.nb_actions, memory=memory,
                     actor=actor, critic=critic, policy_actor=policy_actor, policy_critic=policy_critic,
                     critic_action_input=critic.inputs[2:4], action_type='discrete',
                     policy=policy, test_policy=test_policy, env_details=env_details,
                     batch_size=3, nb_max_steps_recurrent_unrolling=20,
                     nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                     train_interval=5)
print('step2...')
agent.compile([Adam(lr=1e-3, clipnorm=40.), Adam(lr=1e-3, clipnorm=40.)], metrics=['mae'])

chunk_train = np.load('chunk_train.npy')
chunk_train = chunk_train.tolist()
ms = agent.critic_train_on_batch(chunk_train)[0]

# import rl
# rl.util.update_soft_target_model(agent.target_critic, agent.critic, agent.target_model_update)


'''
from rl.util import (clone_model, clone_optimizer,
                     get_soft_target_model_updates, huber_loss,
                     get_object_config)

# make target model
self = agent
self.target_critic = clone_model(critic, self.custom_model_objects)
self.target_critic.compile(optimizer='sgd', loss='mse')

policy_critic.compile(optimizer='sgd', loss='mse')
critic.compile(optimizer='Adam', loss='mse')
policy_actor.compile(optimizer='sgd', loss='mse')
actor.compile(optimizer='sgd', loss='mse')

# define critic update function
discounted_reward = K.placeholder(shape=(3, None, 1))
Qvalue = critic.outputs[0]

loss = huber_loss(y_true=discounted_reward, y_pred=Qvalue, clip_value=np.inf)
loss = K.mean(loss)

# soft target updates
updates = []
updates += critic.optimizer.get_updates(loss=loss, params=critic.trainable_weights)
# updates += get_soft_target_model_updates(self.target_critic, critic, self.target_model_update)
# Update the policy model after every training step.
# updates += get_soft_target_model_updates(self.policy_critic, critic, 1.)
train_fn = K.function(inputs=critic.inputs + [discounted_reward, K.learning_phase()],
                      outputs=[loss],  # [grad_ops, loss]
                      updates=updates)

train_fn(chunk_train)
'''