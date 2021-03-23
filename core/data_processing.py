import argparse
from multiprocessing import Pool

import pandas as pd

from core.feature_extraction import FeatureExtractor, GoalDetector
from core.scenario import Scenario
from core.base import get_data_dir, get_scenario_config_dir
from core.generate_dataset_split import load_dataset_splits


def get_dataset(scenario_name, subset='train', features=True):
    data_set_splits = load_dataset_splits()
    episode_idxes = data_set_splits[scenario_name][subset]
    episode_training_sets = []

    for episode_idx in episode_idxes:
        episode_training_set = pd.read_csv(
            get_data_dir() + '{}_e{}.csv'.format(scenario_name, episode_idx, subset))
        episode_training_set['episode'] = episode_idx
        episode_training_sets.append(episode_training_set)
    training_set = pd.concat(episode_training_sets)
    if features:
        return training_set
    else:
        unique_training_samples = training_set[['episode', 'agent_id', 'initial_frame_id', 'frame_id',
                                            'true_goal', 'true_goal_type', 'fraction_observed']
                                            ].drop_duplicates().reset_index()
        return unique_training_samples


def get_goal_priors(training_set, goal_types, alpha=0):
    agent_goals = training_set[['episode', 'agent_id', 'true_goal', 'true_goal_type']].drop_duplicates()
    print('training_vehicles: {}'.format(agent_goals.shape[0]))
    goal_counts = pd.DataFrame(data=[(x, t, 0) for x in range(len(goal_types)) for t in goal_types[x]],
                               columns=['true_goal', 'true_goal_type', 'goal_count'])

    goal_counts = goal_counts.set_index(['true_goal', 'true_goal_type'])
    goal_counts['goal_count'] += agent_goals.groupby(['true_goal', 'true_goal_type']).size()
    goal_counts = goal_counts.fillna(0)

    # plt.show()
    goal_priors = ((goal_counts.goal_count + alpha) / (agent_goals.shape[0] + alpha * goal_counts.shape[0])).rename('prior')
    goal_priors = goal_priors.reset_index()
    return goal_priors


def prepare_episode_dataset(params):
    scenario_name, episode_idx = params
    print('scenario {} episode {}'.format(scenario_name, episode_idx))
    samples_per_trajectory = 10
    scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
    feature_extractor = FeatureExtractor(scenario.lanelet_map, scenario.config.goal_types)
    episode = scenario.load_episode(episode_idx)

    samples_list = []

    goals = {}  # key: agent id, value: goal idx
    trimmed_trajectories = {}

    # detect goal, and trim trajectory past the goal
    goal_detector = GoalDetector(scenario.config.goals)
    for agent_id, agent in episode.agents.items():
        if agent.agent_type in ['car', 'truck_bus']:
            agent_goals, goal_frames = goal_detector.detect_goals(agent.state_history)
            if len(agent_goals) > 0:
                final_goal_frame_idx = goal_frames[-1] - agent.initial_frame
                trimmed_trajectory = agent.state_history[0:final_goal_frame_idx]
                goals[agent_id] = agent_goals[-1]
                trimmed_trajectories[agent_id] = trimmed_trajectory

    # get features and reachable goals

    for agent_idx, (agent_id, trajectory) in enumerate(trimmed_trajectories.items()):

        print('agent_id {}/{}'.format(agent_idx, len(trimmed_trajectories) - 1))
        # iterate through each sampled point in time for trajectory

        reachable_goals_list = []

        # get reachable goals at each timestep
        for idx in range(0, len(trajectory)):
            goal_routes = feature_extractor.get_goal_routes(trajectory[idx], scenario.config.goals)
            if len([r for r in goal_routes if r is not None]) > 1:
                reachable_goals_list.append(goal_routes)
            else:
                break

        # iterate through "samples_per_trajectory" points
        true_goal_idx = goals[agent_id]
        true_goal_types = scenario.config.goal_types[true_goal_idx]
        if (len(reachable_goals_list) > samples_per_trajectory
                and reachable_goals_list[0][true_goal_idx] is not None):

            # get true goal
            true_goal_loc = scenario.config.goals[true_goal_idx]
            true_goal_route = reachable_goals_list[0][true_goal_idx]
            true_goal_type = feature_extractor.goal_type(trajectory[0], true_goal_loc, true_goal_route,
                                                         true_goal_types)

            step_size = (len(reachable_goals_list) - 1) // samples_per_trajectory
            max_idx = step_size * samples_per_trajectory
            for idx in range(0, max_idx + 1, step_size):
                reachable_goals = reachable_goals_list[idx]
                state = trajectory[idx]
                frames = episode.frames[trajectory[0].frame_id:state.frame_id + 1]

                # iterate through each goal for that point in time
                for goal_idx, route in enumerate(reachable_goals):
                    if route is not None:
                        goal = scenario.config.goals[goal_idx]

                        features = feature_extractor.extract(agent_id, frames, goal, route, goal_idx)

                        sample = features.copy()
                        sample['agent_id'] = agent_id
                        sample['possible_goal'] = goal_idx
                        sample['true_goal'] = true_goal_idx
                        sample['true_goal_type'] = true_goal_type
                        sample['frame_id'] = state.frame_id
                        sample['initial_frame_id'] = trajectory[0].frame_id
                        sample['fraction_observed'] = idx / max_idx
                        samples_list.append(sample)

    samples = pd.DataFrame(data=samples_list)
    samples.to_csv(get_data_dir() + '{}_e{}.csv'.format(scenario_name, episode_idx), index=False)


def main():
    parser = argparse.ArgumentParser(description='Process the dataset')
    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default=None)
    args = parser.parse_args()

    if args.scenario is None:
        scenarios = ['heckstrasse', 'bendplatz', 'frankenberg', 'round']
    else:
        scenarios = [args.scenario]

    params_list = []
    for scenario_name in scenarios:
        scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
        for episode_idx in range(len(scenario.config.episodes)):
            params_list.append((scenario_name, episode_idx))

    with Pool(4) as p:
        p.map(prepare_episode_dataset, params_list)


if __name__ == '__main__':
    main()
