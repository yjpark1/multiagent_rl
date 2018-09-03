import multiagent
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

# load scenario from script
scenario_name = 'simple_speaker_listener'
# scenario_name = 'simple_tag'
scenario_name = 'simple_spread'
scenario = scenarios.load(scenario_name + ".py").Scenario()

# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
env.n

for i in range(5, 3):
    print(i)