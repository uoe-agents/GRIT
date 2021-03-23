import argparse
import json

from core.base import get_dt_config_dir
from decisiontree.dt_goal_recogniser import DecisionTreeGoalRecogniser


def main():
    parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
    parser.add_argument('--scenario', type=str, help='Name of scenario to validate', default=None)
    args = parser.parse_args()

    if args.scenario is None:
        scenario_names = ['heckstrasse', 'bendplatz', 'frankenberg', 'round']
    else:
        scenario_names = [args.scenario]

    for scenario_name in scenario_names:
        with open(get_dt_config_dir() + scenario_name + '.json') as f:
            dt_params = json.load(f)
        model = DecisionTreeGoalRecogniser.train(scenario_name, **dt_params)
        model.save(scenario_name)


if __name__ == '__main__':
    main()
