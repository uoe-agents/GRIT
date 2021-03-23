import numpy as np
import pandas as pd

from core.base import get_data_dir, get_scenario_config_dir
from core.feature_extraction import FeatureExtractor
from core.scenario import Scenario
from goalrecognition.metrics import entropy


class BayesianGoalRecogniser:

    def __init__(self, goal_priors, scenario):
        self.goal_priors = goal_priors
        self.feature_extractor = FeatureExtractor(scenario.lanelet_map,
                                                  scenario.config.goal_types)
        self.scenario = scenario

    def goal_likelihood(self, goal_idx, frames, route, agent_id):
        raise NotImplementedError

    def goal_likelihood_from_features(self, features, goal_type, goal):
        raise NotImplementedError

    def goal_probabilities(self, frames, agent_id):
        state_history = [f.agents[agent_id] for f in frames]
        current_state = state_history[-1]
        goal_routes = self.feature_extractor.get_goal_routes(current_state, self.scenario.config.goals)
        goal_probs = []
        for goal_idx, route in enumerate(goal_routes):
            if route is None:
                goal_prob = 0
            else:
                # get un-normalised "probability"
                goal_types = self.scenario.config.goal_types[goal_idx]
                prior = self.get_goal_prior(goal_idx, state_history[0], route, goal_types)
                if prior == 0:
                    goal_prob = 0
                else:
                    likelihood = self.goal_likelihood(goal_idx, frames, route, agent_id)
                    goal_prob = likelihood * prior
            goal_probs.append(goal_prob)
        goal_probs = np.array(goal_probs)
        goal_probs = goal_probs / goal_probs.sum()
        return goal_probs

    def batch_goal_probabilities(self, dataset):
        """

        Args:
            dataset: DataFrame with columns:
                path_to_goal_length,in_correct_lane,speed,acceleration,angle_in_lane,vehicle_in_front_dist,
                vehicle_in_front_speed,oncoming_vehicle_dist,goal_type,agent_id,possible_goal,true_goal,true_goal_type,
                frame_id,initial_frame_id,fraction_observed

        Returns:

        """
        dataset = dataset.copy()
        model_likelihoods = []

        for index, row in dataset.iterrows():
            features = row[FeatureExtractor.feature_names]
            goal_type = row['goal_type']
            goal = row['possible_goal']
            model_likelihood = self.goal_likelihood_from_features(features, goal_type, goal)

            model_likelihoods.append(model_likelihood)
        dataset['model_likelihood'] = model_likelihoods
        unique_samples = dataset[['episode', 'agent_id', 'frame_id', 'true_goal',
                                  'true_goal_type', 'fraction_observed']].drop_duplicates()
        model_predictions = []
        predicted_goal_types = []
        model_probs = []
        min_probs = []
        max_probs = []
        model_entropys = []
        model_norm_entropys = []
        for index, row in unique_samples.iterrows():
            indices = ((dataset.episode == row.episode)
                       & (dataset.agent_id == row.agent_id)
                       & (dataset.frame_id == row.frame_id))
            goals = dataset.loc[indices][['possible_goal', 'goal_type', 'model_likelihood']]
            goals = goals.merge(self.goal_priors, 'left', left_on=['possible_goal', 'goal_type'],
                                right_on=['true_goal', 'true_goal_type'])
            goals['model_prob'] = goals.model_likelihood * goals.prior
            goals['model_prob'] = goals.model_prob / goals.model_prob.sum()
            idx = goals['model_prob'].idxmax()

            goal_prob_entropy = entropy(goals.model_prob)
            uniform_entropy = entropy(np.ones(goals.model_prob.shape[0])
                                      / goals.model_prob.shape[0])
            norm_entropy = goal_prob_entropy / uniform_entropy
            model_prediction = goals['possible_goal'].loc[idx]
            predicted_goal_type = goals['goal_type'].loc[idx]
            predicted_goal_types.append(predicted_goal_type)
            model_predictions.append(model_prediction)
            model_prob = goals['model_prob'].loc[idx]
            max_prob = goals.model_prob.max()
            min_prob = goals.model_prob.min()
            max_probs.append(max_prob)
            min_probs.append(min_prob)
            model_probs.append(model_prob)
            model_entropys.append(goal_prob_entropy)
            model_norm_entropys.append(norm_entropy)

        unique_samples['model_prediction'] = model_predictions
        unique_samples['predicted_goal_type'] = predicted_goal_types
        unique_samples['model_probs'] = model_probs
        unique_samples['max_probs'] = max_probs
        unique_samples['min_probs'] = min_probs
        unique_samples['model_entropy'] = model_entropys
        unique_samples['model_entropy_norm'] = model_norm_entropys
        return unique_samples

    @classmethod
    def load(cls, scenario_name):
        priors = cls.load_priors(scenario_name)
        scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
        return cls(priors, scenario)

    @staticmethod
    def load_priors(scenario_name):
        return pd.read_csv(get_data_dir() + scenario_name + '_priors.csv')

    def get_goal_prior(self, goal_idx, state, route, goal_types=None):
        goal_loc = self.scenario.config.goals[goal_idx]
        goal_type = self.feature_extractor.goal_type(state, goal_loc, route, goal_types)
        prior_series = self.goal_priors.loc[(self.goal_priors.true_goal == goal_idx) & (self.goal_priors.true_goal_type == goal_type)].prior
        if prior_series.shape[0] == 0:
            return 0
        else:
            return float(prior_series)


class PriorBaseline(BayesianGoalRecogniser):

    def goal_likelihood(self, goal_idx, frames, route, agent_id):
        return 0.5

    def goal_likelihood_from_features(self, features, goal_type, goal):
        return 0.5
