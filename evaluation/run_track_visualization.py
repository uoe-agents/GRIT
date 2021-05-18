"""
Modified version of code from https://github.com/ika-rwth-aachen/drone-dataset-tools
"""

import os
import sys
import glob
import argparse
import pandas as pd

from core.base import get_scenario_config_dir, get_data_dir
from core.scenario import Scenario
from loguru import logger
from core.tracks_import import read_from_csv
from core.track_visualizer import TrackVisualizer
from decisiontree.dt_goal_recogniser import TrainedDecisionTrees, HandcraftedGoalTrees
from goalrecognition.goal_recognition import PriorBaseline


def create_args():
    config_specification = argparse.ArgumentParser(description="ParameterOptimizer")
    # --- Input paths ---
    config_specification.add_argument('--input_path', default="../../inD-dataset/data/",
                                      help="Dir with track files", type=str)
    config_specification.add_argument('--recording_name', default="32",
                                      help="Choose recording name.", type=str)

    config_specification.add_argument('--scenario', default="heckstrasse",
                                      help="Choose scenario name.", type=str)

    config_specification.add_argument('--episode', default=0,
                                      help="Choose an episode ID.", type=int)

    config_specification.add_argument('--goal_recogniser', default=None,
                                      help="Choose goal recognition method.", type=str)

    config_specification.add_argument('--agent_id', default=None,
                                      help="Only show details for this agent", type=int)

    # --- Settings ---
    config_specification.add_argument('--scale_down_factor', default=12,
                                      help="Factor by which the tracks are scaled down to match a scaled down image.",
                                      type=float)
    # --- Visualization settings ---
    config_specification.add_argument('--skip_n_frames', default=5,
                                      help="Skip n frames when using the second skip button.",
                                      type=int)
    config_specification.add_argument('--plotLaneIntersectionPoints', default=False,
                                      help="Optional: decide whether to plot the direction triangle or not.",
                                      type=bool)
    config_specification.add_argument('--plotBoundingBoxes', default=True,
                                      help="Optional: decide whether to plot the bounding boxes or not.",
                                      type=bool)
    config_specification.add_argument('--plotDirectionTriangle', default=True,
                                      help="Optional: decide whether to plot the direction triangle or not.",
                                      type=bool)
    config_specification.add_argument('--plotTrackingLines', default=True,
                                      help="Optional: decide whether to plot the direction lane intersection points or not.",
                                      type=bool)
    config_specification.add_argument('--plotFutureTrackingLines', default=True,
                                      help="Optional: decide whether to plot the tracking lines or not.",
                                      type=bool)
    config_specification.add_argument('--showTextAnnotation', default=True,
                                      help="Optional: decide whether to plot the text annotation or not.",
                                      type=bool)
    config_specification.add_argument('--showClassLabel', default=True,
                                      help="Optional: decide whether to show the class in the text annotation.",
                                      type=bool)
    config_specification.add_argument('--showVelocityLabel', default=True,
                                      help="Optional: decide whether to show the velocity in the text annotation.",
                                      type=bool)
    config_specification.add_argument('--showRotationsLabel', default=False,
                                      help="Optional: decide whether to show the rotation in the text annotation.",
                                      type=bool)
    config_specification.add_argument('--showAgeLabel', default=False,
                                      help="Optional: decide whether to show the current age of the track the text annotation.",
                                      type=bool)

    parsed_config_specification = vars(config_specification.parse_args())
    return parsed_config_specification


if __name__ == '__main__':
    config = create_args()

    scenario = Scenario.load(get_scenario_config_dir() + config['scenario'] + '.json')
    episode = scenario.load_episode(config["episode"])

    episode_dataset = pd.read_csv(
        get_data_dir() + '{}_e{}.csv'.format(config['scenario'], config["episode"]))

    goal_recognisers = {'prior': PriorBaseline,
                        'trained_trees': TrainedDecisionTrees,
                        'handcrafted_trees': HandcraftedGoalTrees}

    if config['goal_recogniser'] is not None:
        goal_recogniser = goal_recognisers[config['goal_recogniser']].load(config['scenario'])
    else:
        goal_recogniser = None

    input_root_path = scenario.config.data_root
    recording_name = scenario.config.episodes[config["episode"]]['recording_id']
    config['input_path'] = input_root_path
    config['recording_name'] = recording_name

    if recording_name is None:
        logger.error("Please specify a recording!")
        sys.exit(1)

    # Search csv files
    tracks_files = glob.glob(input_root_path + recording_name + "*_tracks.csv")
    static_tracks_files = glob.glob(input_root_path + recording_name + "*_tracksMeta.csv")
    recording_meta_files = glob.glob(input_root_path + recording_name + "*_recordingMeta.csv")
    if len(tracks_files) == 0 or len(static_tracks_files) == 0 or len(recording_meta_files) == 0:
        logger.error("Could not find csv files for recording {} in {}. Please check parameters and path!",
                     recording_name, input_root_path)
        sys.exit(1)

    # Load csv files
    logger.info("Loading csv files {}, {} and {}", tracks_files[0], static_tracks_files[0], recording_meta_files[0])
    tracks, static_info, meta_info = read_from_csv(tracks_files[0], static_tracks_files[0], recording_meta_files[0])
    if tracks is None:
        logger.error("Could not load csv files!")
        sys.exit(1)

    # Load background image for visualization
    background_image_path = input_root_path + recording_name + "_background.png"
    if not os.path.exists(background_image_path):
        logger.warning("No background image {} found. Fallback using a black background.".format(background_image_path))
        background_image_path = None
    config["background_image_path"] = background_image_path

    visualization_plot = TrackVisualizer(config, tracks, static_info, meta_info, goal_recogniser=goal_recogniser,
                                         scenario=scenario, episode=episode, agent_id=config["agent_id"],
                                         episode_dataset=episode_dataset)
    visualization_plot.show()
