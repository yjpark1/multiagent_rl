import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import (Input, Conv2D, MaxPooling2D,
                          Concatenate, Bidirectional, LSTM,
                          Dense, TimeDistributed, Flatten,
                          ConvLSTM2D, Conv2DTranspose, GlobalAveragePooling2D,
                          Reshape)
from keras.layers import GRU as RNN
from keras.models import Model


def actor_net(env):
    state_2D = Input(shape=(env.nb_agents, ) + env.state_dim)
    state_1D = Input(shape=(env.nb_agents, ) + (2, ))
    # <actor network>
    # 2d-state encoder
    h_2d = TimeDistributed(Conv2D(32, 4, activation='relu', padding='same', strides=4,
                                  kernel_regularizer=l2(1e-3)))(state_2D)
    h_2d = TimeDistributed(Conv2D(64, 4, activation='relu', padding='same', strides=4,
                                  kernel_regularizer=l2(1e-3)))(h_2d)
    h_2d = TimeDistributed(GlobalAveragePooling2D())(h_2d)
    # combine 1d state & 2d state
    h = Concatenate()([h_2d, state_1D])
    # <multi-agent with RNN>
    h = Dense(64, kernel_regularizer=l2(1e-3))(h)
    h = TimeDistributed(keras.layers.Activation('relu'))(h)
    # LSTM policy network
    # rnn along agents
    h = Bidirectional(RNN(32, kernel_regularizer=l2(1e-3),
                          return_sequences=True))(h)
    h = TimeDistributed(keras.layers.Activation('relu'))(h)
    # action_xy
    h_xy = TimeDistributed(keras.layers.Reshape(target_shape=(8, 8, 1)))(h)
    h_xy = TimeDistributed(Conv2DTranspose(16, 4, activation='relu', padding='same',
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
    actor = Model(inputs=[state_2D, state_1D],
                  outputs=[action_xy, action_type])

    return actor
    

def critic_net(env):
    state_2D = Input(shape=(env.nb_agents, ) + env.state_dim)
    state_1D = Input(shape=(env.nb_agents, ) + (2,))
    # action input
    input_action_xy = Input(shape=(env.nb_agents,  64 * 64,))
    input_action_type = Input(shape=(env.nb_agents, 3,))
    h_action_xy = TimeDistributed(keras.layers.Reshape((64, 64, 1)))(input_action_xy)
    h_2d = Concatenate()([state_2D, h_action_xy])
    # 2d-state encoder
    h_2d = TimeDistributed(Conv2D(32, 4, activation='relu', padding='same', strides=4,
                                  kernel_regularizer=l2(1e-3)))(h_2d)
    h_2d = TimeDistributed(Conv2D(64, 4, activation='relu', padding='same', strides=4,
                                  kernel_regularizer=l2(1e-3)))(h_2d)
    h_2d = TimeDistributed(GlobalAveragePooling2D())(h_2d)
    # combine 1d state & 2d state
    h = Concatenate()([h_2d, state_1D, input_action_type])
    # <multi-agent with RNN>
    # rnn along time steps: need <return sequences = True>
    h = Dense(64, kernel_regularizer=l2(1e-3))(h)
    h = TimeDistributed(keras.layers.Activation('relu'))(h)
    # LSTM policy network
    # rnn along agents and return all outputs of the time steps
    h = Bidirectional(RNN(32, kernel_regularizer=l2(1e-3), return_sequences=False))(h)
    h = keras.layers.Activation('relu')(h)

    # Q value
    q_value = Dense(1, activation='linear', kernel_regularizer=l2(1e-3))(h)
    critic = Model(inputs=[state_2D, state_1D, input_action_xy, input_action_type],
                   outputs=[q_value])

    return critic