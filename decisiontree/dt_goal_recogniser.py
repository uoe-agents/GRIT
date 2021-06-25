import pickle
from sklearn import tree
import numpy as np

from core.base import get_data_dir, get_scenario_config_dir, get_img_dir
from core.data_processing import get_goal_priors, get_dataset
from decisiontree.decision_tree import Node
from core.feature_extraction import FeatureExtractor
from decisiontree.handcrafted_trees import scenario_trees
from core.scenario import Scenario
from goalrecognition.goal_recognition import BayesianGoalRecogniser


class DecisionTreeGoalRecogniser(BayesianGoalRecogniser):

    def __init__(self, goal_priors, scenario, decision_trees):
        super().__init__(goal_priors, scenario)
        self.decision_trees = decision_trees

    def goal_likelihood(self, goal_idx, frames, route, agent_id):
        goal_loc = self.scenario.config.goals[goal_idx]
        features = self.feature_extractor.extract(agent_id, frames, goal_loc, route, goal_idx)
        self.decision_trees[goal_idx][features['goal_type']].reset_reached()
        likelihood = self.decision_trees[goal_idx][features['goal_type']].traverse(features)
        return likelihood

    def goal_likelihood_from_features(self, features, goal_type, goal):
        if goal_type in self.decision_trees[goal]:
            tree = self.decision_trees[goal][goal_type]
            tree_likelihood = tree.traverse(features)
        else:
            tree_likelihood = 0.5
        return tree_likelihood

    @classmethod
    def load(cls, scenario_name):
        priors = cls.load_priors(scenario_name)
        scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
        decision_trees = cls.load_decision_trees(scenario_name)
        return cls(priors, scenario, decision_trees)

    @staticmethod
    def load_decision_trees(scenario_name):
        raise NotImplementedError

    @classmethod
    def train(cls, scenario_name, alpha=1, criterion='gini', min_samples_leaf=1,
              max_leaf_nodes=None, max_depth=None, training_set=None, ccp_alpha=0):
        decision_trees = {}
        scenario = Scenario.load(get_scenario_config_dir() + scenario_name + '.json')
        if training_set is None:
            training_set = get_dataset(scenario_name, subset='train')
        goal_priors = get_goal_priors(training_set, scenario.config.goal_types, alpha=alpha)

        for goal_idx in goal_priors.true_goal.unique():
            decision_trees[goal_idx] = {}
            goal_types = goal_priors.loc[goal_priors.true_goal == goal_idx].true_goal_type.unique()
            for goal_type in goal_types:
                dt_training_set = training_set.loc[(training_set.possible_goal == goal_idx)
                                                   & (training_set.goal_type == goal_type)]
                if dt_training_set.shape[0] > 0:
                    X = dt_training_set[FeatureExtractor.feature_names.keys()].to_numpy()
                    y = (dt_training_set.possible_goal == dt_training_set.true_goal).to_numpy()
                    if y.all() or not y.any():
                        goal_tree = Node(0.5)
                    else:
                        clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,
                            min_samples_leaf=min_samples_leaf, max_depth=max_depth, class_weight='balanced',
                            criterion=criterion, ccp_alpha=ccp_alpha)
                        clf = clf.fit(X, y)
                        goal_tree = Node.from_sklearn(clf, FeatureExtractor.feature_names)
                        goal_tree.set_values(dt_training_set, goal_idx, alpha=alpha)
                else:
                    goal_tree = Node(0.5)

                decision_trees[goal_idx][goal_type] = goal_tree
        return cls(goal_priors, scenario, decision_trees)

    def save(self, scenario_name):
        for goal_idx in self.goal_priors.true_goal.unique():
            goal_types = self.goal_priors.loc[self.goal_priors.true_goal == goal_idx].true_goal_type.unique()
            for goal_type in goal_types:
                goal_tree = self.decision_trees[goal_idx][goal_type]
                pydot_tree = goal_tree.pydot_tree()
                pydot_tree.write_png(get_img_dir() + 'trained_tree_{}_G{}_{}.png'.format(
                    scenario_name, goal_idx, goal_type))
        with open(get_data_dir() + 'trained_trees_{}.p'.format(scenario_name), 'wb') as f:
            pickle.dump(self.decision_trees, f)
        self.goal_priors.to_csv(get_data_dir() + '{}_priors.csv'.format(scenario_name), index=False)


class HandcraftedGoalTrees(DecisionTreeGoalRecogniser):

    @staticmethod
    def load_decision_trees(scenario_name):
        return scenario_trees[scenario_name]


class TrainedDecisionTrees(DecisionTreeGoalRecogniser):

    @staticmethod
    def load_decision_trees(scenario_name):
        with open(get_data_dir() + 'trained_trees_{}.p'.format(scenario_name), 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    depths = []
    for scenario in ['heckstrasse', 'bendplatz', 'frankenberg', 'round']:
        model = TrainedDecisionTrees.load(scenario)
        for goal_type_trees in model.decision_trees.values():
            for tree in goal_type_trees.values():
                depth = tree.get_depth()
                print(depth)
                depths.append(depth)
    print(np.mean(depths))
    print(np.std(depths))
