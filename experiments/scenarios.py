import numpy as np
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv


def local_obs_simple_spread(self, agent, world):
    """
    get positions of all entities in this agent's reference frame
    return local observation
    original return: np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
    """
    entity_pos = []
    for entity in world.landmarks:  # world.entities:
        entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    # entity colors
    entity_color = []
    for entity in world.landmarks:  # world.entities:
        entity_color.append(entity.color)

    return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos)


def local_obs_simple_reference(self, agent, world):
    # goal color
    goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
    if agent.goal_b is not None:
        goal_color[1] = agent.goal_b.color

    # get positions of all entities in this agent's reference frame
    entity_pos = []
    for entity in world.landmarks:
        entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    # entity colors
    entity_color = []
    for entity in world.landmarks:
        entity_color.append(entity.color)
    # communication of all other agents
    comm = []
    for other in world.agents:
        if other is agent: continue
        comm.append(other.state.c)
    return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)


def local_obs_simple_speaker_listener(self, agent, world):
    # goal color
    goal_color = np.zeros(world.dim_color)
    if agent.goal_b is not None:
        goal_color = agent.goal_b.color

    # get positions of all entities in this agent's reference frame
    entity_pos = []
    for entity in world.landmarks:
        entity_pos.append(entity.state.p_pos - agent.state.p_pos)

    # communication of all other agents
    comm = []
    for other in world.agents:
        if other is agent or (other.state.c is None): continue
        comm.append(other.state.c)

    # speaker & listener
    return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color])


def local_obs_multi_speaker_listener(self, agent, world):
    obs = []
    # give listener communication from its speaker
    # obs += [world.speakers[agent.speak_ind].state.c]
    # give listener its own position/velocity,
    obs += [agent.state.p_pos, agent.state.p_vel]

    # give speaker index of their listener
    if hasattr(agent, 'listen_ind'):
        obs += [agent.listen_ind == np.arange(len(world.listeners))]
    else:
        obs += [np.repeat(False, len(world.listeners))]

    # give listener index of their speaker
    if hasattr(agent, 'speak_ind'):
        obs += [agent.speak_ind == np.arange(len(world.speakers))]
    else:
        obs += [np.repeat(False, len(world.speakers))]

    # speaker gets position of listener and goal
    if agent.listener == False:
        obs += [agent.goal_a.state.p_pos, agent.goal_b.state.p_pos]
    else:
        obs += [np.repeat(0., 4)]

    # speaker gets position of listener and goal
    return np.concatenate(obs)


def local_obs_collect_treasure(self, agent, world):
    n_visible = 0  # number of other agents and treasures visible to each agent
    # get positions of all entities in this agent's reference frame
    other_agents = [a.i for a in world.agents if a is not agent]
    closest_agents = sorted(
        zip(world.cached_dist_mag[other_agents, agent.i],
            other_agents))[:n_visible]
    treasures = [t.i for t in self.treasures(world)]
    closest_treasures = sorted(
        zip(world.cached_dist_mag[treasures, agent.i],
            treasures))[:7]

    n_treasure_types = len(world.treasure_types)
    obs = [agent.state.p_pos, agent.state.p_vel]
    # collectors need to know their own state bc it changes
    obs.append((np.arange(n_treasure_types) == agent.holding))
    for _, i in closest_agents:
        a = world.entities[i]
        obs.append(world.cached_dist_vect[i, agent.i])
        obs.append(a.state.p_vel)
        obs.append(self.get_agent_encoding(a, world))
    for _, i in closest_treasures:
        t = world.entities[i]
        obs.append(world.cached_dist_vect[i, agent.i])
        obs.append((np.arange(n_treasure_types) == t.type))

    return np.concatenate(obs)


def make_env(scenario_name, local_observation=True,
             benchmark=False, discrete_action=True):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    
    simple_spread
    simple_reference
    simple_speaker_listener
    collect_treasure
    multi_speaker_listener

    '''
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    if local_observation:
        if scenario_name == 'simple_spread':
            scenario.observation = local_obs_simple_spread.__get__(scenario)
        elif scenario_name == 'simple_reference':
            scenario.observation = local_obs_simple_reference.__get__(scenario)
        elif scenario_name == 'simple_speaker_listener':
            scenario.observation = local_obs_simple_speaker_listener.__get__(scenario)
        elif scenario_name == 'multi_speaker_listener':
            # scenario.observation = local_obs_multi_speaker_listener.__get__(scenario)
            print('origin')
        elif scenario_name == 'fullobs_collect_treasure':
            scenario.observation = local_obs_collect_treasure.__get__(scenario)
        else:
            print('error: unsupported scenario!')

    # create world
    world = scenario.make_world()
    world.collaborative = False  # to get individual reward

    # create multiagent environment
    if hasattr(scenario, 'post_step'):
        post_step = scenario.post_step
    else:
        post_step = None
    if benchmark:
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            post_step_callback=post_step,
                            info_callback=scenario.benchmark_data,
                            discrete_action=discrete_action)
    else:
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            post_step_callback=post_step,
                            discrete_action=discrete_action)
    env.force_discrete_action = True
    return env


if __name__ == '__main__':
    from keras.utils import to_categorical
    import time

    sc = ['simple_spread', 'simple_reference', 'simple_speaker_listener',
          'fullobs_collect_treasure', 'multi_speaker_listener']

    env = make_env(scenario_name=sc[2], local_observation=False)
    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)

    actions = [x.sample() for x in env.action_space]
    a = env.action_space[1]

    actions = to_categorical(actions, num_classes=5)
    # env.reset()
    # s, r, d, _ = env.step(actions)
    s = env.reset()

