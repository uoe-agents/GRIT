import numpy as np
import pandas as pd

from core.base import get_scenario_config_dir, get_data_dir
from core.data_processing import get_dataset
from decisiontree.dt_goal_recogniser import HandcraftedGoalTrees, TrainedDecisionTrees
from goalrecognition.goal_recognition import PriorBaseline
from core.scenario import Scenario

scenario_name = 'heckstrasse'
scenario = Scenario.load(get_scenario_config_dir() + scenario_name + '.json')
print('loading episodes')
episodes = scenario.load_episodes()

models = {'prior_baseline': PriorBaseline,
          'handcrafted_trees': HandcraftedGoalTrees,
          'trained_trees': TrainedDecisionTrees}

dataset_names = ['train', 'valid', 'test']

accuracies = pd.DataFrame(index=models.keys(), columns=dataset_names)
cross_entropies = pd.DataFrame(index=models.keys(), columns=dataset_names)

for dataset_name in dataset_names:
    dataset = get_dataset(scenario_name, dataset_name, features=False)
    predictions = {}
    num_goals = len(scenario.config.goals)
    targets = dataset.true_goal.to_numpy()

    for model_name, model in models.items():
        model = model.load(scenario_name)
        model_predictions = []

        for index, sample in dataset.iterrows():
            print('{}/{} samples'.format(index + 1, dataset.shape[0]))
            frames = episodes[sample.episode].frames[sample.initial_frame_id:sample.frame_id+1]
            goal_probabilities = model.goal_probabilities(frames, sample.agent_id)
            model_predictions.append(goal_probabilities)
        model_predictions = np.array(model_predictions)
        accuracy = (dataset.true_goal.to_numpy() == np.argmax(model_predictions, axis=1)).mean()
        log_predictions = np.log(model_predictions)
        target_predictions = model_predictions[np.arange(model_predictions.shape[0]), targets]

        cross_entropy = -np.mean(np.log(target_predictions[target_predictions!=0]))

        accuracies.loc[model_name, dataset_name] = accuracy
        cross_entropies.loc[model_name, dataset_name] = cross_entropy

        np.save(get_data_dir() + '{}_{}_predictions.npy'.format(model_name, dataset_name), model_predictions)

        print('{} accuracy: {:.3f}'.format(model_name, accuracy))
        print('{} cross entropy: {:.3f}'.format(model_name, cross_entropy))

print('accuracy:')
print(accuracies)
print('cross entropy:')
print(cross_entropies)
