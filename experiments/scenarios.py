import numpy as np
import multiagent.scenarios as scenarios


def local_observation(agent, world):
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

scenario_name = 'simple_spread'

def make_scenario(scenario_name, local_obs=False):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # change to local observation
    if local_obs:
        scenario.observation = local_observation
    
    scenario.reward = reward_simple_spread

    return scenario