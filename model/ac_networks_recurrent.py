import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import (Input, Conv2D, MaxPooling2D,
                          Concatenate, Bidirectional, LSTM,
                          Dense, TimeDistributed, Flatten,
                          ConvLSTM2D, Conv2DTranspose, GlobalAveragePooling2D,
                          Reshape)
from keras.layers import CuDNNGRU as LSTM
from keras.models import Model


def actor_net(env, model_type, batch_size):
    state_2D = Input(batch_shape=(batch_size, env.nb_agents, None) + env.state_dim)
    state_1D = Input(batch_shape=(batch_size, env.nb_agents, None) + (2, ))
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
    h = TimeDistributed(LSTM(64, return_sequences=True, stateful=True,
                             kernel_regularizer=l2(1e-3)))(h)

    if model_type == 'train_model':
        h = TimeDistributed(TimeDistributed(keras.layers.Activation('relu')))(h)
        # LSTM policy network
        # rnn along agents
        h = keras.layers.Permute((2, 1, 3))(h)
        h = TimeDistributed(Bidirectional(LSTM(32, kernel_regularizer=l2(1e-3), 
                                               return_sequences=True)))(h)
        h = keras.layers.Permute((2, 1, 3))(h)
        h = TimeDistributed(TimeDistributed(keras.layers.Activation('relu')))(h)
        # action_xy
        h_xy = TimeDistributed(TimeDistributed(keras.layers.Reshape(target_shape=(8, 8, 1))))(h)
        h_xy = TimeDistributed(TimeDistributed(Conv2DTranspose(16, 4, activation='relu', padding='same',
                                                               strides=(2, 2),
                                                               kernel_regularizer=l2(1e-3))))(h_xy)
        h_xy = TimeDistributed(TimeDistributed(Conv2DTranspose(1, 4, activation='linear', padding='same',
                                                               strides=(4, 4),
                                                               kernel_regularizer=l2(1e-3))))(h_xy)
        h_xy = TimeDistributed(TimeDistributed(Flatten()))(h_xy)
        # h_xy = TimeDistributed(TimeDistributed(Reshape((4096,))))(h_xy)
        action_xy = TimeDistributed(TimeDistributed(keras.layers.Activation('softmax')))(h_xy)

    elif model_type == 'policy_model':
        h = keras.layers.Lambda(lambda x: x[:, :, -1, :])(h)
        h = TimeDistributed(keras.layers.Activation('relu'))(h)
        # LSTM policy network
        # rnn along agents
        h = Bidirectional(LSTM(32, kernel_regularizer=l2(1e-3),
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
    

def critic_net(env, model_type, batch_size):    
    state_2D = Input(batch_shape=(batch_size, env.nb_agents, None) + env.state_dim)
    state_1D = Input(batch_shape=(batch_size, env.nb_agents, None) + (2,))
    # action input
    input_action_xy = Input(batch_shape=(batch_size, env.nb_agents, None, 64 * 64,))
    input_action_type = Input(batch_shape=(batch_size, env.nb_agents, None, 3,))
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
    h = TimeDistributed(LSTM(64, return_sequences=True, stateful=True, 
                             kernel_regularizer=l2(1e-3)))(h)

    if model_type == 'train_model':
        h = TimeDistributed(TimeDistributed(keras.layers.Activation('relu')))(h)
        # LSTM policy network
        # rnn along agents and return all outputs of the time steps
        h = keras.layers.Permute((2, 1, 3))(h)
        h = TimeDistributed(Bidirectional(LSTM(32, kernel_regularizer=l2(1e-3), 
                                               return_sequences=False)))(h)
        h = TimeDistributed(keras.layers.Activation('relu'))(h)

    elif model_type == 'policy_model':
        h = keras.layers.Lambda(lambda x: x[:, :, -1, :])(h)
        h = TimeDistributed(keras.layers.Activation('relu'))(h)
        # LSTM policy network
        # rnn along agents
        h = Bidirectional(LSTM(32, kernel_regularizer=l2(1e-3), 
                               return_sequences=True))(h)
        h = TimeDistributed(keras.layers.Activation('relu'))(h)
        
    # Q value
    q_value = Dense(1, activation='linear', kernel_regularizer=l2(1e-3))(h)
    critic = Model(inputs=[state_2D, state_1D, input_action_xy, input_action_type],
                   outputs=[q_value])

    return critic