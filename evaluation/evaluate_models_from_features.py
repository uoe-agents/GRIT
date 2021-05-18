import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from core.base import get_scenario_config_dir
from core.data_processing import get_dataset
from decisiontree.dt_goal_recogniser import TrainedDecisionTrees
from goalrecognition.goal_recognition import PriorBaseline
from core.scenario import Scenario


def main():
    plt.style.use('ggplot')

    parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
    parser.add_argument('--scenario', type=str, help='Name of scenario to validate', default=None)
    args = parser.parse_args()

    if args.scenario is None:
        scenario_names = ['heckstrasse', 'bendplatz', 'frankenberg', 'round']
    else:
        scenario_names = [args.scenario]

    # print('loading episodes')
    # episodes = scenario.load_episodes()

    models = {'prior_baseline': PriorBaseline,
              #'handcrafted_trees': HandcraftedGoalTrees,
              'trained_trees': TrainedDecisionTrees}

    accuracies = pd.DataFrame(index=models.keys(), columns=scenario_names)
    accuracies_sem = pd.DataFrame(index=models.keys(), columns=scenario_names)
    cross_entropies = pd.DataFrame(index=models.keys(), columns=scenario_names)
    entropies = pd.DataFrame(index=models.keys(), columns=scenario_names)
    norm_entropies = pd.DataFrame(index=models.keys(), columns=scenario_names)
    avg_max_prob = pd.DataFrame(index=models.keys(), columns=scenario_names)
    avg_min_prob = pd.DataFrame(index=models.keys(), columns=scenario_names)

    predictions = {}
    dataset_name = 'test'

    for scenario_name in scenario_names:
        dataset = get_dataset(scenario_name, dataset_name)
        scenario = Scenario.load(get_scenario_config_dir() + scenario_name + '.json')
        dataset_predictions = {}
        num_goals = len(scenario.config.goals)
        targets = dataset.true_goal.to_numpy()

        for model_name, model in models.items():
            model = model.load(scenario_name)
            unique_samples = model.batch_goal_probabilities(dataset)
            unique_samples['model_correct'] = (unique_samples['model_prediction']
                                               == unique_samples['true_goal'])
            cross_entropy = -np.mean(np.log(unique_samples.loc[
                                                        unique_samples.model_probs != 0, 'model_probs']))
            accuracy = unique_samples.model_correct.mean()
            accuracies_sem.loc[model_name, scenario_name] = unique_samples.model_correct.sem()
            accuracies.loc[model_name, scenario_name] = accuracy
            cross_entropies.loc[model_name, scenario_name] = cross_entropy
            entropies.loc[model_name, scenario_name] = unique_samples.model_entropy.mean()
            norm_entropies.loc[model_name, scenario_name] = unique_samples.model_entropy_norm.mean()
            avg_max_prob.loc[model_name, scenario_name] = unique_samples.max_probs.mean()
            avg_min_prob.loc[model_name, scenario_name] = unique_samples.min_probs.mean()
            dataset_predictions[model_name] = unique_samples
            print('{} accuracy: {:.3f}'.format(model_name, accuracy))
            print('{} cross entropy: {:.3f}'.format(model_name, cross_entropy))

        predictions[scenario_name] = dataset_predictions

    print('accuracy:')
    print(accuracies)
    print('accuracy sem:')
    print(accuracies_sem)
    print('\ncross entropy:')
    print(cross_entropies)
    print('\nentropy:')
    print(entropies)
    print('\nnormalised entropy:')
    print(norm_entropies)
    print('\naverage max probability:')
    print(avg_max_prob)
    print('\naverage min probability:')
    print(avg_min_prob)

    for scenario_name in scenario_names:

        fig, ax = plt.subplots()
        for model_name, model in models.items():
            unique_samples = predictions[scenario_name][model_name]
            fraction_observed_grouped = unique_samples[['model_correct', 'fraction_observed']].groupby('fraction_observed')
            accuracy = fraction_observed_grouped.mean()
            accuracy_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())
            accuracy.rename(columns={'model_correct': model_name}).plot(ax=ax)
            plt.fill_between(accuracy_sem.index, (accuracy + accuracy_sem).model_correct.to_numpy(),
                             (accuracy - accuracy_sem).model_correct.to_numpy(), alpha=0.2)
        plt.xlabel('fraction of trajectory observed')
        plt.title('Accuracy ({})'.format(scenario_name))
        plt.ylim([0, 1])
        plt.show()

        fig, ax = plt.subplots()
        for model_name, model in models.items():
            unique_samples = predictions[scenario_name][model_name]
            fraction_observed_grouped = unique_samples[['model_entropy', 'fraction_observed']].groupby('fraction_observed')
            entropy_norm = fraction_observed_grouped.mean()
            entropy_norm_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())
            entropy_norm.rename(columns={'model_entropy': model_name}).plot(ax=ax)
            plt.fill_between(entropy_norm_sem.index, (entropy_norm + entropy_norm_sem).model_entropy.to_numpy(),
                             (entropy_norm - entropy_norm_sem).model_entropy.to_numpy(), alpha=0.2)
        plt.xlabel('fraction of trajectory observed')
        plt.title('Normalised Entropy ({})'.format(scenario_name))
        plt.ylim([0, 1])
        plt.show()


if __name__ == '__main__':
    main()
