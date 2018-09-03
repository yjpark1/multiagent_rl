# -*- coding: utf-8 -*-
# RLServer.environment
# combined with keras-rl
import numpy as np
import msgpack
from model import GlobalVariable as gvar
import collections
from util.CustomLog import cLogger

logger = cLogger.getLogger()


# env_details = {'ally': ['verture']*2, 'enemy': ['zealot']*3, 'state_dim': (64, 64, 3 + 2)}

class StarCraftEnvironment(object):
    """Add class docstring."""
    reward_range = (-np.inf, np.inf)
    action_space = [np.arange(64 * 64), np.arange(3)]
    observation_space = None

    def __init__(self, agent_name, env_details):
        # intialize
        self.prev_health_ally = 0
        self.prev_health_enemy = 0

        self.agent_name = agent_name
        self.size_of_state_list_aggregating = 1
        self.state_list_aggregating = collections.deque([],
                                                        maxlen=self.size_of_state_list_aggregating)
        self.env_details = env_details
        self.num_ally = len(self.env_details['ally'])
        self.num_enemy = len(self.env_details['enemy'])
        self.state_dim = self.env_details['state_dim']
        self.nb_agents = self.num_ally
        self.nb_actions = (64 * 64, 3)

        # restart flag
        self.flag_restart = 0
        self.prev_flag_restart = False

        # minimap size
        self.size_minimap = 64

        # defalut health
        self.default_health_ally = 40 * len(self.env_details['ally'])
        self.default_health_enemy = 35 * len(self.env_details['enemy'])

        # current number of ally units
        self.num_ally_current = self.num_ally
        self.num_enemy_current = self.num_enemy

        # reference coordinate
        self.reference_coor = self._make_reference_coordinate()

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        [get_next_state in StarCraftEnvironment]
        # Arguments
            action (object): An action provided by the environment.

        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        # add step
        self.nb_step += 1

        # get restart flag
        self.flag_restart = int(gvar.flag_restart)
        # print(self.nb_step, ': ', self.flag_restart)
        # set action
        action_bwapi = self._make_action_bwapi(action)
        action_token = self._make_action_token(action_bwapi)
        gvar.release_action = False
        gvar.action = action_token

        # a -> s, r, d
        token_refine = self._refine_token()
        next_state = self._get_state(token_refine)
        reward = self._get_reward()
        done = self._get_done()
        info = dict()

        # stop rl agent
        gvar.service_flag = 0

        # log
        if done:
            logger.info('reward: ' + str(self.R) + ' num enemy: ' + str(self.num_enemy_current) +
                        ' num ally: ' + str(self.num_ally_current))

        return next_state, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        [get_initial_state in StarCraftEnvironment]
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        ## for debug
        self.R = 0
        self.hist = []
        ##

        self.nb_step = 1
        self.flag_restart = 0
        gvar.release_action = True
        self.num_ally_current = self.num_ally
        self.num_enemy_current = self.num_enemy

        token_refine = self._refine_token()
        initial_state = self._get_state(token_refine)

        return initial_state

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)

        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        pass

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()


    def _make_reference_coordinate(self):
        base_x = np.arange(-32, 32 + 1)
        base_x = base_x[base_x != 0]
        base_y = np.arange(32, -32 - 1, -1)
        base_y = base_y[base_y != 0]

        base_coor = np.zeros((64, 64, 2))
        for i, y in enumerate(base_y):
            for j, x in enumerate(base_x):
                base_coor[i, j, 0] = x
                base_coor[i, j, 1] = y

        return base_coor


    def _make_action_bwapi(self, action):
        action_bwapi = []
        # for each agent
        for a_xy, a_type in zip(action[0], action[1]):
            a_xy = np.where(np.arange(64 * 64).reshape(64, 64) == a_xy)

            # use reference coordinate
            a_xy = self.reference_coor[a_xy[0], a_xy[1], :]
            a_xy = np.squeeze(a_xy)

            a_x = int(a_xy[0] * 4)
            a_y = int(a_xy[1] * 4)
            # [x, y, nothing/attack/move]
            a_type = int(a_type)
            a = [a_x, a_y, a_type]
            action_bwapi.append(a)

        return action_bwapi


    def _make_action_token(self, action_bwapi):
        action_token = msgpack.packb(action_bwapi)
        return action_token


    def _deserialization(self):
        while True:
            if len(gvar.token_deque) == 0:
                pass
            else:
                token = gvar.token_deque.pop()
                break

        # print(token)
        token = msgpack.unpackb(token, raw=False)
        ##
        # print('raw token: ', token)
        ##
        return token


    def _parse_token(self, token):
        token_minimap = token[0:2]
        token_resource = token[-1]
        ##
        # print(token_resource)
        ##
        token_unit = token[2:-1]

        return (token_minimap, token_unit, token_resource)


    def _str2num(self, token_minimap, token_unit, token_resource):
        # make numpy array
        token_minimap = np.stack(
            [np.array(token_minimap[0]).reshape(64, 64), np.array(token_minimap[1]).reshape(64, 64)], axis=-1)
        token_minimap = token_minimap.astype(dtype='float32')
        token_unit = np.array([np.array(x, dtype='float32') for x in token_unit])
        token_resource = np.array(token_resource, dtype='float32')

        return (token_minimap, token_unit, token_resource)


    def _calibrate_token_unit(self):
        if len(self.token_unit) == 0:
            self.token_unit = np.zeros(shape=(self.num_ally + self.num_enemy, 7))
            self.token_unit[:self.num_ally, 0] = 0  # assign ally
            self.token_unit[self.num_ally:, 0] = 1  # assign enemy

        # current units
        numAlly = sum(self.token_unit[:, 0] == 0)
        numEnemy = sum(self.token_unit[:, 0] == 1)

        # dead units
        numDeadAlly = self.num_ally - numAlly
        numDeadEnemy = self.num_enemy - numEnemy

        # calibration
        if numDeadAlly > 0:
            add = np.zeros((numDeadAlly, 7))
            self.token_unit = np.vstack([self.token_unit, add])

        if numDeadEnemy > 0:
            add = np.zeros((numDeadEnemy, 7))
            add[:, 0] = 1
            self.token_unit = np.vstack([self.token_unit, add])


    def _2DArrayfromUnitToken(self, token_unit, state_minimap):
        '''
        <input info>
        BWAPI: for a unit, vector = [isEnemy, HP, Sheild, Cooldown, X, Y, UnitType]
        token_unit = np.array([unit1, unit2, ..., unitN])
        unitN = np.array([isEnemy, HP, Sheild, Cooldown, X, Y, UnitType])

        <output info>
        *For now, we use only cooldown because we play homogeneous minigame*
        out = np.array([unit1, unit2, ..., unitN])
        unitN = (2D np.arrray, 1D np.array)

        * 2D np.arrray: X, Y coordinate
            # 1st channel: ally HP
            # 2nd channel: enemy HP
            # 3rd channel: enemy Sheild

        * 1D np.array: cooldown
        '''
        # use isEnemy

        # TODO : does it guarantee that bwapi send agents information with same order every time?!!
        token_unit_ally = token_unit[token_unit[:, 0] == 0]
        states_unit_2D = []

        for idx, my_unit in enumerate(token_unit_ally):
            state_ally = np.zeros(shape=self.state_dim) # 64, 64, 5(3+2)
            xy_my_unit = my_unit[4:6]

            ####################################################################################################
            # TODO : append to pad value -1 out of range
            xy_my_unit_for_padding = np.ceil(xy_my_unit / 8.)

            # print('xy_my_unit_for_padding : {}'.format(xy_my_unit_for_padding))

            # x padding!!
            x_left_padding = np.max([(xy_my_unit_for_padding[0] - 32) * -1, 0])
            x_right_buffer = 0

            if x_left_padding == 0:
                x_right_buffer = np.max([(xy_my_unit_for_padding[0] + 32) - 256, 0])
                if x_right_buffer > 0:
                    for i in range(3):
                        for w in range(64 - int(x_right_buffer), 64, 1):
                            state_ally[w, :, i] = -1
            else:
                for i in range(3):
                    for w in range(int(x_left_padding)):
                        state_ally[w, :, i] = -1

            # print('x_left_padding : {}, x_right_buffer : {}'.format(x_left_padding, x_right_buffer))

            # y padding!!
            y_up_buffer = np.max([(xy_my_unit_for_padding[1] - 32) * -1, 0])

            y_bottom_buffer = 0
            if y_up_buffer == 0:
                y_bottom_buffer = np.max([xy_my_unit_for_padding[1] + 32 - 256, 0])
                if y_bottom_buffer > 0:
                    for i in range(3):
                        for w in range(64 - int(y_bottom_buffer), 64, 1):
                            state_ally[:, w, i] = -1
            else:
                for i in range(3):
                    for w in range(int(y_up_buffer)):
                        state_ally[:, w, i] = -1
            ####################################################################################################

            for other_unit in np.delete(token_unit, idx, 0):
                xy_other_unit = other_unit[4:6]

                # walkposition scale relative distance
                xy_relative = (xy_other_unit - xy_my_unit) / 8.
                xy_relative = np.ceil(xy_relative)
                xy_relative = np.where((self.reference_coor[:, :, 0] == xy_relative[0]) *
                                       (self.reference_coor[:, :, 1] == xy_relative[1]))
                xy_relative = np.squeeze(np.array(xy_relative, dtype=int))

                if xy_relative.size != 0:  # xy coordinate out of range (64, 64)
                    # assign values
                    if other_unit[0] == 0:
                        # assign ally HP
                        state_ally[xy_relative[0], xy_relative[1], 0] += other_unit[1]

                    elif other_unit[0] == 1:
                        # assign enemy HP
                        state_ally[xy_relative[0], xy_relative[1], 1] += other_unit[1]
                        # assign enemy Shield
                        state_ally[xy_relative[0], xy_relative[1], 2] += other_unit[2]

            # insert minimap
            state_ally[:, :, 3:5] = state_minimap

            states_unit_2D.append(state_ally)
        states_unit_2D = np.array(states_unit_2D)
        states_unit_1D = token_unit_ally[:, [1, 3]]

        return states_unit_2D, states_unit_1D


    def _refine_token(self):
        token = self._deserialization()
        token_minimap, token_unit, token_resource = self._parse_token(token)

        tokens = self._str2num(token_minimap, token_unit, token_resource)
        state_minimap, token_resource = tokens[0], tokens[2]

        # to use for reward
        self.token_unit = tokens[1]

        # unit state
        # check dead ally to calibrate state
        self._calibrate_token_unit()
        state_unit_2D, state_unit_1D = self._2DArrayfromUnitToken(self.token_unit, state_minimap)

        # resource: mineral, gas
        self.mineral = token_resource[0]  # win count
        self.gas = token_resource[1]  # game count

        return state_unit_2D, state_unit_1D


    def _get_state(self, token_refine):
        state_unit_2D = token_refine[0]
        state_unit_1D = token_refine[1]

        # make state: combine (minimap + state unit 2D) + 1D
        state = [state_unit_2D, state_unit_1D]

        return state

    def _get_Health(self):
        # isEnemy
        ally = self.token_unit[self.token_unit[:, 0] == 0]
        enemy = self.token_unit[self.token_unit[:, 0] == 1]

        self.currentHealth_ally = sum(ally[:, 1])
        self.currentHealth_enemy = sum(enemy[:, 1]) + sum(enemy[:, 2])

    # _get_reward 함수와 _get_done 함수는 서로 얽혀 있음. 수정 시 주의요
    def _get_reward(self):
        '''
        health = hp(1) + sheild(2)
        '''
        # get health
        self._get_Health()
        currentHealth_ally = self.currentHealth_ally
        currentHealth_enemy = self.currentHealth_enemy

        numAlly = sum(self.token_unit[self.token_unit[:, 1] > 0, 0] == 0)
        numEnemy = sum(self.token_unit[self.token_unit[:, 1] > 0, 0] == 1)

        reward = 0
        if (self.num_ally_current - numAlly) > 0:
            reward += -10 * (self.num_ally_current - numAlly)

        if (self.num_enemy_current - numEnemy) > 0:
            reward += 10 * (self.num_enemy_current - numEnemy)

        self.num_ally_current = numAlly
        self.num_enemy_current = numEnemy

        # reward by health change
        delta_ally = self.prev_health_ally - currentHealth_ally
        delta_enemy = self.prev_health_enemy - currentHealth_enemy

        # scaling delta
        delta_ally = 1 * delta_ally / self.default_health_ally
        delta_enemy = 1 * delta_enemy / self.default_health_enemy

        reward += delta_enemy - delta_ally

        if currentHealth_ally == 0 or currentHealth_enemy == 0:
            # 다 끝난 경우
            reward = 0

        elif (currentHealth_ally == self.default_health_ally and
              currentHealth_enemy == self.default_health_enemy):
            # 새로 시작한 경우
            reward = 0

        ## for debug
        self.R += reward

        return reward

    def _get_done(self):
        done = False
        currentHealth_ally = self.currentHealth_ally
        currentHealth_enemy = self.currentHealth_enemy

        # test1 = (self.currentHealth_ally - self.prev_health_ally) >= (self.default_health_ally - 8)
        numAlly = sum(self.token_unit[self.token_unit[:, 1] > 0, 0] == 0)
        numEnemy = sum(self.token_unit[self.token_unit[:, 1] > 0, 0] == 1)

        if (numAlly == 0 or numEnemy == 0) and self.nb_step > 10:
            done = True

        if self.flag_restart == 1:
            done = True

        # update prev_health_ally
        self.prev_health_ally = currentHealth_ally
        self.prev_health_enemy = currentHealth_enemy

        return done


