# policy for multi agent rl
from __future__ import division
import numpy as np

from rl.util import *
from rl.policy import Policy

class starcraft_multiagent_eGreedyPolicy(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    nb_actions = (64*64, 3)
    """
    def __init__(self, nb_agents, nb_actions, eps=.1):
        super(starcraft_multiagent_eGreedyPolicy, self).__init__()
        self.nb_agents = nb_agents
        self.nb_actions = nb_actions
        self.eps = eps

    def select_action(self, preference):
        """Return the selected action

        # Arguments
            q_values (list): [action_xy (np.array), action_type (np.array)]
            [(1, nb_agents, actions), (1, nb_agents, actions)]

        # Returns
            Selection action: [(x,y), nothing/attack/move]
            [(nb_agents, 1), (nb_agents, 1)]
        """
        assert len(preference) == 2
        preference_xy = np.squeeze(preference[0])
        preference_type = np.squeeze(preference[1])

        actions_xy = []
        actions_type = []
        for agent in range(self.nb_agents):
            # select action_xy
            if np.random.uniform() < self.eps:
                action_xy = np.random.random_integers(0, self.nb_actions[0]-1)
            else:
                action_xy = np.argmax(preference_xy[agent])
            actions_xy.append(action_xy)

            # select action_type
            if np.random.uniform() < self.eps:
                action_type = np.random.random_integers(0, self.nb_actions[1]-1)
            else:
                action_type = np.argmax(preference_type[agent])
            actions_type.append(action_type)

        actions_xy = np.array(actions_xy)
        actions_type = np.array(actions_type)

        return [actions_xy, actions_type]

    def get_config(self):
        """Return configurations of EpsGreedyPolicy

        # Returns
            Dict of config
        """
        config = super(starcraft_multiagent_eGreedyPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class MA_EpsGreedyQPolicy(Policy):
    def __init__(self, eps=.1):
        super(MA_EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        assert q_values.ndim == 3
        q_values = np.squeeze(q_values, axis=0)
        nb_agents = q_values.shape[0]
        nb_actions = q_values.shape[1]

        actions = []
        for agent in range(nb_agents):
            if np.random.uniform() < self.eps:
                action = np.random.random_integers(0, nb_actions-1)
            else:
                action = np.argmax(q_values[agent])
            actions.append(action)
        actions = np.array(actions)
        return actions

    def get_config(self):
        config = super(MA_EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class MA_GreedyQPolicy(Policy):
    def select_action(self, q_values):
        assert q_values.ndim == 3
        q_values = np.squeeze(q_values, axis=0)
        actions = np.argmax(q_values, axis=-1)
        return actions


class MA_BoltzmannQPolicy(Policy):
    def __init__(self, tau=1., clip=(-500., 500.)):
        super(MA_BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        assert q_values.ndim == 3
        q_values = np.squeeze(q_values, axis=0)
        actions = np.apply_along_axis(self.select_action_agent, -1, q_values)
        return actions

    def select_action_agent(self, q_value):
        assert q_value.ndim == 1
        q_value = q_value.astype('float64')
        nb_actions = q_value.shape[0]

        exp_value = np.exp(np.clip(q_value / self.tau, self.clip[0], self.clip[1]))
        probs = exp_value / np.sum(exp_value)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        config = super(MA_BoltzmannQPolicy, self).get_config()
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config


class MA_MaxBoltzmannQPolicy(Policy):
    """
    A combination of the eps-greedy and Boltzman q-policy.

    Wiering, M.: Explorations in Efficient Reinforcement Learning.
    PhD thesis, University of Amserdam, Amsterdam (1999)

    https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf
    """
    def __init__(self, eps=.1, tau=1., clip=(-500., 500.)):
        super(MA_MaxBoltzmannQPolicy, self).__init__()
        self.eps = eps
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        assert q_values.ndim == 3
        q_values = np.squeeze(q_values, axis=0)
        actions = np.apply_along_axis(self.select_action_agent, -1, q_values)
        return actions

    def select_action_agent(self, q_value):
        assert q_value.ndim == 1
        q_value = q_value.astype('float64')
        nb_actions = q_value.shape[0]

        if np.random.uniform() < self.eps:
            exp_value = np.exp(np.clip(q_value / self.tau, self.clip[0], self.clip[1]))
            probs = exp_value / np.sum(exp_value)
            action = np.random.choice(range(nb_actions), p=probs)
        else:
            action = np.argmax(q_value)
        return action

    def get_config(self):
        config = super(MA_MaxBoltzmannQPolicy, self).get_config()
        config['eps'] = self.eps
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config

