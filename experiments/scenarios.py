import numpy as np
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv


def local_obs_simple_spread(agent, world):
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


def local_obs_collect_treasure(self, agent, world):
    print('local!')
    n_visible = 7  # number of other agents and treasures visible to each agent
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
    if agent.collector:
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


def reward_simple_spread(self, agent, world):
    """
    Agents are rewarded based on minimum agent distance to each landmark, 
    penalized for collisions
    return shared reward & individual reward 
    """
    rew = 0
    for l in world.landmarks:
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        rew -= min(dists)
    
    # check collision between my agent & other agents 
    if agent.collide:
        for a in world.agents:
            if self.is_collision(a, agent) & (agent != a):
                rew -= 1

    return rew

# simple_spread
# simple_reference
# simple_speaker_listener
# collect_treasure
# multi_speaker_listener

# scenario_name = 'simple_spread'
def make_env(scenario_name, benchmark=False, discrete_action=True):
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
    '''
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
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
    return env


if __name__ == '__main__':
    # scenario_name = 'fullobs_collect_treasure'
    # scenario_name = 'simple_spread'
    # scenario_name = 'simple_reference'
    # scenario_name = 'simple_speaker_listener'
    scenario_name = 'multi_speaker_listener'


    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # scenario.observation = local_obs_collect_treasure.__get__(scenario)
    world = scenario.make_world()
    world.collaborative = False  # to get individual reward

    # <create multi-agent environment>
    if hasattr(scenario, 'post_step'):
        post_step = scenario.post_step
    else:
        post_step = None

    env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward,
                        observation_callback=scenario.observation,
                        post_step_callback=post_step,
                        discrete_action=True)

    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)

    actions = [x.sample() for x in env.action_space]

    from keras.utils import to_categorical
    actions = to_categorical(actions, num_classes=5)
    env.reset()
    s, r, d, _ = env.step(actions)
    env.shared_reward
    # env.close()








