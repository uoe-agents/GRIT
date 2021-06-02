import numpy as np
import json
import imageio
import matplotlib.pyplot as plt
import lanelet2
from lanelet2 import geometry
from lanelet2.core import BasicPoint2d

from core import map_vis_lanelet2
from core.tracks_import import read_from_csv


class ScenarioConfig:
    """Metadata about a scenario used for goal recognition"""

    def __init__(self, config_dict):
        self.config_dict = config_dict

    @classmethod
    def load(cls, file_path):
        """Loads the scenario metadata into from a json file

        Args:
            file_path (str): path to the file to load

        Returns:
            ScenarioConfig: metadata about the scenario

        """
        with open(file_path) as f:
            scenario_meta_dict = json.load(f)
        return cls(scenario_meta_dict)

    @property
    def goals(self):
        """list of [int, int]: Possible goals for agents in this scenario"""
        return self.config_dict.get('goals')

    @property
    def name(self):
        """str: Name of the scenario"""
        return self.config_dict.get('name')

    @property
    def goal_types(self):
        """list of list of str: Possible goals for agents in this scenario"""
        return self.config_dict.get('goal_types')

    @property
    def lanelet_file(self):
        """str: Path to the *.osm file specifying the lanelet2 map"""
        return self.config_dict.get('lanelet_file')

    @property
    def lat_origin(self):
        """float: Latitude of the origin"""
        return self.config_dict.get('lat_origin')

    @property
    def lon_origin(self):
        """float: Longitude of the origin"""
        return self.config_dict.get('lon_origin')

    @property
    def data_format(self):
        """str: Format in which the data is stored"""
        return self.config_dict.get('data_format')

    @property
    def data_root(self):
        """str: Path to directory in which the data is stored"""
        return self.config_dict.get('data_root')

    @property
    def episodes(self):
        """list of dict: Configuration for all episodes for this scenario"""
        return self.config_dict.get('episodes')

    @property
    def background_image(self):
        """str: Path to background image"""
        return self.config_dict.get('background_image')

    @property
    def background_px_to_meter(self):
        """float: Pixels per meter in background image"""
        return self.config_dict.get('background_px_to_meter')

    @property
    def reachable_pairs(self):
        """list: Pairs of points, where the second point should be reachable from the first
           To can be used for validating maps"""
        return self.config_dict.get('reachable_pairs')


class EpisodeConfig:
    """Metadata about an episode"""

    def __init__(self, config):
        self.config = config

    @property
    def recording_id(self):
        """str: unique id identifying the episode"""
        return self.config.get('recording_id')


class Agent:

    def __init__(self, state_history, metadata):
        self.state_history = state_history
        self.agent_id = metadata.agent_id
        self.width = metadata.width
        self.length = metadata.length
        self.agent_type = metadata.agent_type
        self.initial_frame = metadata.initial_frame
        self.final_frame = metadata.final_frame
        self.num_frames = metadata.final_frame - metadata.initial_frame + 1
        self.parked = geometry.distance(self.state_history[0].point, self.state_history[-1].point) < 1

    def plot_trajectory(self, *args, **kwargs):
        x = [s.x for s in self.state_history]
        y = [s.y for s in self.state_history]
        plt.plot(x, y, *args, **kwargs)


class AgentMetadata:

    def __init__(self, agent_id, width, length, agent_type, initial_frame, final_frame):
        self.agent_id = agent_id
        self.width = width
        self.length = length
        self.agent_type = agent_type
        self.initial_frame = initial_frame
        self.final_frame = final_frame


class AgentState:

    def __init__(self, frame_id, x, y, v_x, v_y, heading, a_x, a_y,
                 v_lon, v_lat, a_lon, a_lat):
        self.frame_id = frame_id
        self.x = x
        self.y = y
        self.v_x = v_x
        self.v_y = v_y
        self.heading = heading
        self.a_x = a_x
        self.a_y = a_y
        self.v_lon = v_lon
        self.v_lat = v_lat
        self.a_lon = a_lon
        self.a_lat = a_lat

    @property
    def point(self):
        return BasicPoint2d(self.x, self.y)

    def plot(self):
        plt.plot(self.x, self.y, 'yo')


class Frame:
    def __init__(self, frame_id):
        self.frame_id = frame_id
        self.agents = {}

    def add_agent_state(self, agent_id, state):
        self.agents[agent_id] = state


class Episode:

    def __init__(self, agents, frames):
        self.agents = agents
        self.frames = frames


class EpisodeLoader:

    def __init__(self, scenario_config):
        self.scenario_config = scenario_config

    def load(self, recording_id):
        raise NotImplementedError


class IndEpisodeLoader(EpisodeLoader):

    def load(self, config):
        track_file = self.scenario_config.data_root \
            + '/{}_tracks.csv'.format(config.recording_id)
        static_tracks_file = self.scenario_config.data_root \
            + '/{}_tracksMeta.csv'.format(config.recording_id)
        recordings_meta_file = self.scenario_config.data_root \
            + '/{}_recordingMeta.csv'.format(config.recording_id)
        tracks, static_info, meta_info = read_from_csv(
            track_file, static_tracks_file, recordings_meta_file)

        num_frames = round(meta_info['frameRate'] * meta_info['duration']) + 1

        agents = {}
        frames = [Frame(i) for i in range(num_frames)]

        for track_meta in static_info:
            agent_meta = self._agent_meta_from_track_meta(track_meta)

            state_history = []
            track = tracks[agent_meta.agent_id]
            num_agent_frames = agent_meta.final_frame - agent_meta.initial_frame + 1
            for idx in range(num_agent_frames):
                state = self._state_from_tracks(track, idx)
                state_history.append(state)
                frames[state.frame_id].add_agent_state(agent_meta.agent_id, state)
            agent = Agent(state_history, agent_meta)
            agents[agent_meta.agent_id] = agent

        return Episode(agents, frames)

    @staticmethod
    def _state_from_tracks(track, idx):
        heading = np.deg2rad(track['heading'][idx])
        heading = np.unwrap([0, heading])[1]
        return AgentState(track['frame'][idx],
                   track['xCenter'][idx],
                   track['yCenter'][idx],
                   track['xVelocity'][idx],
                   track['yVelocity'][idx],
                   heading,
                   track['xAcceleration'][idx],
                   track['yAcceleration'][idx],
                   track['lonVelocity'][idx],
                   track['latVelocity'][idx],
                   track['lonAcceleration'][idx],
                   track['latAcceleration'][idx])

    @staticmethod
    def _agent_meta_from_track_meta(track_meta):
        return AgentMetadata(track_meta['trackId'],
                             track_meta['width'],
                             track_meta['length'],
                             track_meta['class'],
                             track_meta['initialFrame'],
                             track_meta['finalFrame'])


class EpisodeLoaderFactory:

    episode_loaders = {'ind': IndEpisodeLoader}

    @classmethod
    def get_loader(cls, scenario_config):
        loader = cls.episode_loaders[scenario_config.data_format]
        if loader is None:
            raise ValueError('Invalid data format')
        return loader(scenario_config)


class Scenario:
    def __init__(self, config):
        self.config = config
        self.lanelet_map = self.load_lanelet_map()

    def load_lanelet_map(self):
        origin = lanelet2.io.Origin(self.config.lat_origin, self.config.lon_origin)
        projector = lanelet2.projection.UtmProjector(origin)
        lanelet_map, err_list = lanelet2.io.loadRobust(self.config.lanelet_file, projector)
        # for error in err_list:
        #     print(error)
        return lanelet_map

    def load_episodes(self):
        loader = EpisodeLoaderFactory.get_loader(self.config)
        episodes = []
        for idx, c in enumerate(self.config.episodes):
            print('loading episode {}/{}'.format(idx+1, len(self.config.episodes)))
            episode = loader.load(EpisodeConfig(c))
            episodes.append(episode)
        return episodes

    def load_episode(self, episode_id):
        loader = EpisodeLoaderFactory.get_loader(self.config)
        return loader.load(EpisodeConfig(self.config.episodes[episode_id]))

    @classmethod
    def load(cls, file_path):
        config = ScenarioConfig.load(file_path)
        return cls(config)

    def plot(self):
        axes = plt.gca()
        map_vis_lanelet2.draw_lanelet_map(self.lanelet_map, axes)

        # plot background image
        if self.config.background_image is not None:
            background_path = self.config.data_root + self.config.background_image
            background = imageio.imread(background_path)
            rescale_factor = self.config.background_px_to_meter
            extent = (0, int(background.shape[1] * rescale_factor),
                      -int(background.shape[0] * rescale_factor), 0)
            plt.imshow(background, extent=extent)
            plt.xlim([extent[0], extent[1]])
            plt.ylim([extent[2], extent[3]])

        self.plot_goals(axes)

    def plot_goals(self, axes, scale=1, flipy=False):
        # plot goals
        goal_locations = self.config.goals
        for idx, g in enumerate(goal_locations):
            x = g[0] / scale
            y = g[1] / scale * (1-2*int(flipy))
            circle = plt.Circle((x, y), 1.5/scale, color='r')
            axes.add_artist(circle)
            label = 'G{}'.format(idx)
            axes.annotate(label, (x, y), color='white')
