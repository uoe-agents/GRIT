import matplotlib.pyplot as plt
import argparse
from core.base import get_scenario_config_dir
from core.scenario import Scenario


parser = argparse.ArgumentParser(description='create a plot of lanelets and goals')
parser.add_argument('--scenario', type=str, default='bendplatz')
parser.add_argument('--plot_traj', action='store_true')
args = parser.parse_args()

scenario_name = args.scenario
scenario = Scenario.load(get_scenario_config_dir() + scenario_name + '.json')
scenario.plot()

if args.plot_traj:
    episodes = scenario.load_episodes()
    for episode in episodes:
        for agent in episode.agents.values():
            if agent.agent_type == 'car':
                x = [s.x for s in agent.state_history]
                y = [s.y for s in agent.state_history]
                plt.plot(x, y)
plt.show()
