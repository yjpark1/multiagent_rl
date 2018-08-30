from rl import util

from tornado import gen
from tornado.websocket import websocket_connect
from tornado.websocket import WebSocketClosedError
from tornado.ioloop import IOLoop

import tensorflow as tf
from keras import backend as K
# from rl.agents.asyncAgents.async_a2c import DiscreteA2CAgent

import warnings
from copy import deepcopy
from keras.callbacks import History

import numpy as np
import logging as logger
import pickle

from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)

APPLICATION_JSON = 'application/json'
DEFAULT_CONNECT_TIMEOUT = 60
DEFAULT_REQUEST_TIMEOUT = 60

class runLocalAsyncAgent:
    '''
    this class has the same function as <rl.core.agent.fit> method
    '''

    def __init__(self, agent, url, env, nb_steps, get_global_network_interval=3,
                 action_repetition=1, callbacks=None, verbose=2,
                 visualize=False, nb_max_start_steps=0, start_step_policy=None,
                 log_interval=10000, nb_max_episode_steps=None,
                 global_newtork_ip='localhost', global_newtowrk_port=9044, timeout=5):
        # super(runLocalAsyncAgent, self).__init__()

        # agent argument
        self.agent = agent

        # fit argument
        self.env = env
        self.nb_steps = nb_steps
        self.action_repetition = action_repetition
        self.callbacks = callbacks
        self.verbose = verbose
        self.visualize = visualize
        self.nb_max_start_steps = nb_max_start_steps
        self.start_step_policy = start_step_policy
        self.log_interval = log_interval
        self.nb_max_episode_steps = nb_max_episode_steps

        # cummunication argument
        self.ws = None
        self.ws_recv = None
        self.is_connection_closed = False

        self.is_received_msg = False
        self.received_weight = None

        self.get_global_network_interval = get_global_network_interval

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.url = url

        self.ioloop = IOLoop.instance()
        self.connect()

        self.ioloop.start()

    @gen.coroutine
    def cb_on_message(self, weight):
        # print('we get the parameters from server : {}'.format(weight))
        util.set_weight_with_serialized_data(self.agent.actor, self.agent.critic, weight)

    @gen.coroutine
    def connect(self):
        print("trying to connect")
        isConnected = False

        while isConnected is False:
            try:
                self.ws = yield websocket_connect(self.url, on_message_callback=self.cb_on_message)
                isConnected = True
            except Exception as e:
                print("connection error : {}".format(e))
                isConnected = False

        print("connected")
        self.fit(env=self.env, nb_steps=self.nb_steps,
                 action_repetition=self.action_repetition,
                 callbacks=self.callbacks, verbose=self.verbose,
                 visualize=self.visualize, nb_max_start_steps=self.nb_max_start_steps,
                 start_step_policy=self.start_step_policy, log_interval=self.log_interval,
                 nb_max_episode_steps=self.nb_max_episode_steps)

    # update policy network and value network every episode
    @gen.coroutine
    def send_gradient_to_global_agent(self, actor_grad, critic_grad):
        try:
            grads = (actor_grad, critic_grad)
            yield self.ws.write_message(pickle.dumps(grads), binary=True)

        except WebSocketClosedError:
            logger.warning("WS_CLOSED", "Could Not send Message: " + 'sending gradient failed...')
            # Send Websocket Closed Error to Paired Opponent
            self.close()

    @gen.coroutine
    def get_weight_from_global_network(self):
        logger.debug('get_weight_from_global_network :: trying to get message from global network!!')
        while True:
            if self.is_connection_closed is True:
                self.connect()
            else:
                break

        yield self.ws.write_message('send')
        yield gen.sleep(0.1)

    @property
    def metrics_names(self):
        # return self.critic.metrics_names[:] + ['mean_action']
        return ['critic_loss', 'actor_loss']

    @gen.coroutine
    def backward(self, reward, terminal=False):
        metrics = [np.nan, np.nan]

        # Clip the reward to be in reward_range and perform book-keeping.
        # TODO: reward clipping using <process_reward in Process>
        self.agent.reward_accumulator.append(reward)
        self.agent.terminal_accumulator.append(terminal)

        perform_train = (self.agent.training is True) and (terminal is True)
        if not perform_train:
            return metrics

        episode_steps = len(self.agent.reward_accumulator)

        assert episode_steps == len(self.agent.observation_accumulator)
        assert episode_steps == len(self.agent.terminal_accumulator)
        assert episode_steps == len(self.agent.action_accumulator)

        # Accumulate data for gradient computation.
        observations = self.agent.process_state_batch(self.agent.observation_accumulator)

        # TODO: make bootstrapping compatbile with LSTMs
        Vs, Rs = self.agent.discount_rewards(observations)

        # Remove latest value, which we have no use for.
        # observations = np.array(observations[:-1])
        # actions = np.array(self.agent.action_accumulator[:-1])
        # rewards = np.array(self.agent.reward_accumulator[:-1])
        # terminals = np.array(self.agent.terminal_accumulator[:-1])
        # Rs = np.array(Rs[:-1])
        # Vs = np.array(Vs[:-1])

        observations = np.array(observations)
        actions = np.array(self.agent.action_accumulator)
        rewards = np.array(self.agent.reward_accumulator)
        terminals = np.array(self.agent.terminal_accumulator)
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
        assert actions.shape == (episode_steps, self.agent.nb_actions)
        assert terminals.shape == (episode_steps,)

        # Update critic. This also updates the local critic in the process.
        output_critic = self.agent.critic_train_on_batch([observations, Rs])
        critic_metrics, critic_grad = output_critic[0], output_critic[1]

        advantages = Rs - Vs
        output_actor = self.agent.actor_train_on_batch([observations, actions, advantages])
        actor_metrics, actor_grad = output_actor[0], output_actor[1]

        metrics = [critic_metrics, actor_metrics]
        print(metrics)

        self.send_gradient_to_global_agent(critic_grad, actor_grad)

        # Reset state for next update round. We keep the latest data point around since we haven't
        # used it.
        self.agent.observation_accumulator = [self.agent.observation_accumulator[-1]]
        self.agent.action_accumulator = [self.agent.action_accumulator[-1]]
        self.agent.terminal_accumulator = [self.agent.terminal_accumulator[-1]]
        self.agent.reward_accumulator = [self.agent.reward_accumulator[-1]]

        return metrics

    @gen.coroutine
    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        """Trains the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.agent.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.agent.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self.agent._on_train_begin()
        callbacks.on_train_begin()

        episode = 0
        self.step = 0
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = 0
                    episode_reward = 0.

                    # Obtain the initial observation by resetting the environment.
                    self.agent.reset_states()
                    observation = deepcopy(env.reset())
                    if self.agent.processor is not None:
                        observation = self.agent.processor.process_observation(observation)
                    assert observation is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.agent.processor is not None:
                            action = self.agent.processor.process_action(action)
                        callbacks.on_action_begin(action)
                        observation, reward, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.agent.processor is not None:
                            observation, reward, done, info = self.agent.processor.process_step(observation, reward, done,
                                                                                          info)
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn(
                                'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                                    nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.agent.processor is not None:
                                observation = self.agent.processor.process_observation(observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.agent.forward(observation)
                if self.agent.processor is not None:
                    action = self.agent.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.agent.processor is not None:
                        observation, r, done, info = self.agent.processor.process_step(observation, r, done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward += r
                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics = yield self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.agent.forward(observation)
                    yield self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None

                    # TODO: yield ??
                    if episode % self.get_global_network_interval == 0:
                        yield self.get_weight_from_global_network()


        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self.agent._on_train_end()

        return history

    def close(self):
        self.is_connection_closed = True

    def keep_alive(self):
        if self.is_connection_closed:
            print('keey_alive : disconnected')
            self.connect()
        else:
            print('keey_alive : send heart bit message to check ')
            self.ws.write_message("check")

'''
if __name__ == "__main__":
    localAgent = runLocalAsyncAgent("ws://localhost:9044?local_agent_id=2", timeout=50000)
'''