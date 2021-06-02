import numpy as np
import lanelet2
from lanelet2.core import BasicPoint2d, BoundingBox2d
from lanelet2 import geometry

from core.lanelet_helpers import LaneletHelpers
from core.scenario import AgentState


class FeatureExtractor:
    feature_names = {'path_to_goal_length': 'scalar',
                     'in_correct_lane': 'binary',
                     'speed': 'scalar',
                     'acceleration': 'scalar',
                     'angle_in_lane': 'scalar',
                     'vehicle_in_front_dist': 'scalar',
                     'vehicle_in_front_speed': 'scalar',
                     'oncoming_vehicle_dist': 'scalar',
                     'oncoming_vehicle_speed': 'scalar'}

    def __init__(self, lanelet_map, goal_types=None):
        self.lanelet_map = lanelet_map
        self.traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)
        self.routing_graph = lanelet2.routing.RoutingGraph(lanelet_map, self.traffic_rules)
        self.goal_types = goal_types

    def extract(self, agent_id, frames, goal, route, goal_idx=None):
        """Extracts a dict of features
        """

        current_frame = frames[-1]
        current_state = current_frame.agents[agent_id]
        initial_state = frames[0].agents[agent_id]
        current_lanelet = route.shortestPath()[0]

        if route is None:
            route = self.route_to_goal(current_lanelet, goal)
        if route is None:
            raise ValueError('Unreachable goal')

        speed = current_state.v_lon
        acceleration = current_state.a_lon
        in_correct_lane = self.in_correct_lane(route)
        path_to_goal_length = self.path_to_goal_length(current_state, goal, route)
        angle_in_lane = self.angle_in_lane(current_state, current_lanelet)

        goal_types = None if goal_idx is None or self.goal_types is None else self.goal_types[goal_idx]
        goal_type = self.goal_type(initial_state, goal, route, goal_types)

        vehicle_in_front_id, vehicle_in_front_dist = self.vehicle_in_front(agent_id, route, current_frame)
        if vehicle_in_front_id is None:
            vehicle_in_front_speed = 20
            vehicle_in_front_dist = 100
        else:
            vehicle_in_front = current_frame.agents[vehicle_in_front_id]
            vehicle_in_front_speed = vehicle_in_front.v_lon

        oncoming_vehicle_id, oncoming_vehicle_dist = self.oncoming_vehicle(agent_id, route, current_frame)
        if oncoming_vehicle_id is None:
            oncoming_vehicle_speed = 20
        else:
            oncoming_vehicle_speed = current_frame.agents[oncoming_vehicle_id].v_lon
        #
        # if agent_id == 15 and oncoming_vehicle_dist < 0 and goal_idx==3:
        #     import pdb; pdb.set_trace()
        #     self.oncoming_vehicle(agent_id, route, current_frame)

        return {'path_to_goal_length': path_to_goal_length,
                'in_correct_lane': in_correct_lane,
                'speed': speed,
                'acceleration': acceleration,
                'angle_in_lane': angle_in_lane,
                'vehicle_in_front_dist': vehicle_in_front_dist,
                'vehicle_in_front_speed': vehicle_in_front_speed,
                'oncoming_vehicle_dist': oncoming_vehicle_dist,
                'oncoming_vehicle_speed': oncoming_vehicle_speed,
                'goal_type': goal_type}

    @staticmethod
    def angle_in_lane(state: AgentState, lanelet):
        """
        Get the signed angle between the vehicle heading and the lane heading
        Args:
            state: current state of the vehicle
            lanelet: : current lanelet of the vehicle

        Returns: angle in radians
        """
        lane_heading = LaneletHelpers.heading_at(lanelet, state.point)
        angle_diff = np.diff(np.unwrap([lane_heading, state.heading]))[0]
        return angle_diff

    def route_to_goal(self, start_lanelet, goal):
        goal_point = BasicPoint2d(goal[0], goal[1])
        end_lanelets = self.lanelets_at(goal_point)
        best_route = None
        for end_lanelet in end_lanelets:
            route = self.routing_graph.getRoute(start_lanelet, end_lanelet)
            if (route is not None and route.shortestPath()[-1] == end_lanelet
                    and (best_route is None or route.length2d() < best_route.length2d())):
                best_route = route
        return best_route

    def get_current_lanelet(self, state, previous_lanelet=None):
        point = state.point
        radius = 3
        bounding_box = BoundingBox2d(BasicPoint2d(point.x - radius, point.y - radius),
                                     BasicPoint2d(point.x + radius, point.y + radius))
        nearby_lanelets = self.lanelet_map.laneletLayer.search(bounding_box)

        best_lanelet = None
        best_angle_diff = None
        best_dist = None
        best_can_pass = False
        for lanelet in nearby_lanelets:
            if self.traffic_rules.canPass(lanelet):
                dist_from_point = geometry.distance(lanelet, point)
                angle_diff = abs(self.angle_in_lane(state, lanelet))
                can_pass = (False if previous_lanelet is None
                            else self.can_pass(previous_lanelet, lanelet))
                if (angle_diff < np.pi / 2
                        and (best_lanelet is None
                             or (can_pass and not best_can_pass)
                             or ((can_pass or not best_can_pass)
                                 and (dist_from_point < best_dist
                                      or (best_dist == dist_from_point
                                          and angle_diff < best_angle_diff))))):
                    best_lanelet = lanelet
                    best_angle_diff = angle_diff
                    best_dist = dist_from_point
                    best_can_pass = can_pass
        return best_lanelet

    def lanelets_at(self, point):
        nearest_lanelets = geometry.findWithin2d(self.lanelet_map.laneletLayer, point)
        matching_lanelets = []
        for distance, lanelet in nearest_lanelets:
            if distance == 0 and self.traffic_rules.canPass(lanelet):
                matching_lanelets.append(lanelet)
        return matching_lanelets

    def get_goal_routes(self, state: AgentState, goals):
        """
            get most likely current lanelet and corresponding route for each goal

            lanelet must be:
                * close (radius 3m)
                * direction within 90 degrees of car (reduce this?)
                * can pass
                * goal is reachable from

            Rank lanelets based on:
                * distance to current point
                * angle diff
        """
        point = state.point
        radius = 3
        bounding_box = BoundingBox2d(BasicPoint2d(point.x - radius, point.y - radius),
                                     BasicPoint2d(point.x + radius, point.y + radius))
        nearby_lanelets = [l for l in self.lanelet_map.laneletLayer.search(bounding_box)
                           if len(l.centerline) > 0]

        angle_diffs = [abs(self.angle_in_lane(state, l)) for l in nearby_lanelets]
        dists_from_point = [geometry.distance(l, point) for l in nearby_lanelets]

        # filter out invalid lanelets
        possible_lanelets = []
        for idx, lanelet in enumerate(nearby_lanelets):
            if (angle_diffs[idx] < np.pi / 4
                    and dists_from_point[idx] < radius
                    and self.traffic_rules.canPass(lanelet)):
                possible_lanelets.append(idx)

        # find best lanelet for each goal
        goal_lanelets = []
        goal_routes = []
        for goal in goals:
            # find reachable lanelets for each goal
            best_idx = None
            best_route = None
            for lanelet_idx in possible_lanelets:
                if (best_idx is None
                    or dists_from_point[lanelet_idx] < dists_from_point[best_idx]
                    or (dists_from_point[lanelet_idx] == dists_from_point[best_idx]
                        and angle_diffs[lanelet_idx] < angle_diffs[best_idx])):
                    lanelet = nearby_lanelets[lanelet_idx]
                    route = self.route_to_goal(lanelet, goal)
                    if route is not None:
                        best_idx = lanelet_idx
                        best_route = route
            if best_idx is None:
                goal_lanelet = None
            else:
                goal_lanelet = nearby_lanelets[best_idx]

            goal_lanelets.append(goal_lanelet)
            goal_routes.append(best_route)

        return goal_routes

    @staticmethod
    def get_vehicles_in_route(ego_agent_id, route, frame):
        path = route.shortestPath()
        agents = []
        for agent_id, agent in frame.agents.items():
            if agent_id != ego_agent_id:
                for lanelet in path:
                    if geometry.inside(lanelet, agent.point):
                        agents.append(agent_id)
        return agents

    def get_lanelet_sequence(self, states):
        # get the correspoding lanelets for a sequence of frames
        lanelets = []
        lanelet = None
        for state in states:
            lanelet = self.get_current_lanelet(state, lanelet)
            lanelets.append(lanelet)
        return lanelets

    def can_pass(self, a, b):
        # can we legally pass directly from lanelet a to b
        return (a == b or LaneletHelpers.follows(b, a)
                or self.traffic_rules.canChangeLane(a, b))

    def lanelet_at(self, point):
        lanelets = self.lanelets_at(point)
        if len(lanelets) == 0:
            return None
        return lanelets[0]

    @staticmethod
    def in_correct_lane(route):
        path = route.shortestPath()
        return len(path) == len(path.getRemainingLane(path[0]))

    @classmethod
    def path_to_goal_length(cls, state, goal, route):
        end_point = BasicPoint2d(*goal)
        return cls.path_to_point_length(state, end_point, route)

    @classmethod
    def vehicle_in_front(cls, ego_agent_id, route, frame):
        state = frame.agents[ego_agent_id]
        vehicles_in_route = cls.get_vehicles_in_route(ego_agent_id, route, frame) # TODO discount ego agent
        min_dist = np.inf
        vehicle_in_front = None

        path = route.shortestPath()

        ego_dist_along = LaneletHelpers.dist_along_path(path, state.point)

        # find vehicle in front with closest distance
        for agent_id in vehicles_in_route:
            agent_point = frame.agents[agent_id].point
            agent_dist = LaneletHelpers.dist_along_path(path, agent_point)
            dist = agent_dist - ego_dist_along
            if 1e-4 < dist < min_dist:
                vehicle_in_front = agent_id
                min_dist = dist

        return vehicle_in_front, min_dist

    @staticmethod
    def path_to_point_length(state, point, route):
        path = route.shortestPath()
        end_lanelet = path[-1]
        end_lanelet_dist = LaneletHelpers.dist_along(end_lanelet, point)

        start_point = BasicPoint2d(state.x, state.y)
        start_lanelet = path[0]
        start_lanelet_dist = LaneletHelpers.dist_along(start_lanelet, start_point)

        dist = end_lanelet_dist - start_lanelet_dist
        if len(path) > 1:
            prev_lanelet = start_lanelet
            for idx in range(len(path) - 1):
                lanelet = path[idx]
                lane_change = (prev_lanelet.leftBound == lanelet.rightBound
                               or prev_lanelet.rightBound == lanelet.leftBound)
                if not lane_change:
                    dist += geometry.length2d(lanelet)
        return dist

    @staticmethod
    def angle_to_goal(state, goal):
        goal_heading = np.arctan2(goal[1] - state.y, goal[0] - state.x)
        return np.diff(np.unwrap([goal_heading, state.heading]))[0]

    def following_lanelets(self, lanelet, max_depth=10):
        lanelets = []
        if max_depth > 0:
            for following_lanelet in self.routing_graph.following(lanelet):
                lanelets.append(lanelet)
                lanelets.extend(self.following_lanelets(following_lanelet, max_depth-1))
        return lanelets

    def lanelets_to_cross(self, route):
        # get higher priority lanelets to cross
        # TODO: may need testing on new scenarios (other than heckstrasse)
        path = route.shortestPath()
        lanelets_to_cross = []
        crossing_points = []

        following_lanelets = self.following_lanelets(path[0])

        crossed_line = False
        for path_lanelet in path:

            if self.is_yield_lanelet(path_lanelet):
                crossed_line = True

            if not crossed_line:
                crossed_line_lanelet, crossing_point = self.lanelet_crosses_line(path_lanelet)
                if crossed_line_lanelet is not None and crossed_line_lanelet not in following_lanelets:
                    lanelets_to_cross.append(crossed_line_lanelet)
                    crossing_points.append(crossing_point)
                    crossed_line = True

            if crossed_line:
                # check if merged
                if len(self.routing_graph.previous(path_lanelet)) > 1:
                    crossed_line = False
                else:
                    for lanelet in self.lanelet_map.laneletLayer:
                        if lanelet not in following_lanelets:
                            crossing_point = self.lanelet_crosses_lanelet(path_lanelet, lanelet)
                            if crossing_point is not None:
                                lanelets_to_cross.append(lanelet)
                                crossing_points.append(crossing_point)

        return lanelets_to_cross, crossing_points

    def oncoming_vehicle(self, ego_agent_id, route, frame, max_dist=100):
        oncoming_vehicles = self.oncoming_vehicles(ego_agent_id, route, frame, max_dist)
        min_dist = max_dist
        closest_vehicle_id = None
        for agent_id, (agent, dist) in oncoming_vehicles.items():
            if dist < min_dist:
                min_dist = dist
                closest_vehicle_id = agent_id
        return closest_vehicle_id, min_dist

    def oncoming_vehicles(self, ego_agent_id, route, frame, max_dist=100):
        # get oncoming vehicles in lanes to cross
        oncoming_vehicles = {}
        lanelets_to_cross, crossing_points = self.lanelets_to_cross(route)
        for lanelet, point in zip(lanelets_to_cross, crossing_points):
            lanelet_start_dist = LaneletHelpers.dist_along(lanelet, point)
            lanelet_oncoming_vehicles = self.lanelet_oncoming_vehicles(
                ego_agent_id, frame, lanelet, lanelet_start_dist, max_dist)

            for agent_id, (agent, dist) in lanelet_oncoming_vehicles.items():
                if agent_id not in oncoming_vehicles or dist < oncoming_vehicles[agent_id][1]:
                    oncoming_vehicles[agent_id] = (agent, dist)

        return oncoming_vehicles

    def lanelet_oncoming_vehicles(self, ego_agent_id, frame, lanelet, lanelet_start_dist, max_dist):
        # get vehicles oncoming to a root lanelet
        oncoming_vehicles = {}
        for agent_id, agent in frame.agents.items():
            if (agent_id != ego_agent_id
                    and geometry.inside(lanelet, agent.point)
                    and abs(self.angle_in_lane(agent, lanelet) <= np.pi/4)
                    ):
                dist_along_lanelet = LaneletHelpers.dist_along(lanelet, agent.point)
                total_dist_along = lanelet_start_dist - dist_along_lanelet
                if total_dist_along < max_dist:
                    oncoming_vehicles[agent_id] = (agent, total_dist_along)

        if lanelet_start_dist < max_dist:
            for prev_lanelet in self.routing_graph.previous(lanelet):
                if self.traffic_rules.canPass(prev_lanelet):
                    prev_lanelet_start_dist = lanelet_start_dist + geometry.length2d(prev_lanelet)
                    prev_oncoming_vehicles = self.lanelet_oncoming_vehicles(
                        ego_agent_id, frame, prev_lanelet, prev_lanelet_start_dist, max_dist)

                    for agent_id, (agent, dist) in prev_oncoming_vehicles.items():
                        if agent_id not in oncoming_vehicles or dist < oncoming_vehicles[agent_id][1]:
                            oncoming_vehicles[agent_id] = (agent, dist)

        return oncoming_vehicles

    def lanelet_crosses_lanelet(self, path_lanelet, lanelet):
        # check if a lanelet crosses another lanelet, return overlap centroid
        if path_lanelet != lanelet and self.traffic_rules.canPass(lanelet):
            overlap_area, centroid = LaneletHelpers.overlap_area(path_lanelet, lanelet)
            split = (self.routing_graph.previous(path_lanelet)
                     == self.routing_graph.previous(lanelet))
            if overlap_area > 1 and not split:
                return centroid
        return None

    def previous_lanelets(self, lanelet):
        lanelets = []
        for x in range(10):
            prev_lanelets = self.routing_graph.previous(lanelet)
            if len(prev_lanelets) == 1:
                lanelet = prev_lanelets[0]
                lanelets.append(lanelet)
            else:
                break
        return lanelets

    def is_yield_lanelet(self, lanelet):
        for reg in self.lanelet_map.regulatoryElementLayer:
            if lanelet in reg.yieldLanelets():
                return True
        return False

    def lanelet_crosses_line(self, path_lanelet):
        # Check if the midline of a lanelet crosses a road marking.
        # Used for getting priority lanelets to cross

        prev_path_lanelets = self.routing_graph.previous(path_lanelet)
        prev_path_lanelet = prev_path_lanelets[0] if len(prev_path_lanelets) == 1 else None

        for lanelet in self.lanelet_map.laneletLayer:

            prev_lanelets = self.previous_lanelets(lanelet)

            if (path_lanelet != lanelet
                    and prev_path_lanelet not in prev_lanelets
                    and self.traffic_rules.canPass(lanelet)):
                left_virtual = ('type' in lanelet.leftBound.attributes
                                and lanelet.leftBound.attributes['type'] == 'virtual')
                right_virtual = ('type' in lanelet.rightBound.attributes
                                 and lanelet.rightBound.attributes['type'] == 'virtual')
                path_centerline = geometry.to2D(path_lanelet.centerline)
                right_bound = geometry.to2D(lanelet.rightBound)
                left_bound = geometry.to2D(lanelet.leftBound)
                left_intersects = (not left_virtual and
                                   geometry.intersects2d(path_centerline, left_bound))
                right_intersects = (not right_virtual and
                                    geometry.intersects2d(path_centerline, right_bound))
                if left_intersects:
                    intersection_point = LaneletHelpers.intersection_point(
                        path_centerline, left_bound)
                    return lanelet, intersection_point
                elif right_intersects:
                    intersection_point = LaneletHelpers.intersection_point(
                        path_centerline, right_bound)
                    return lanelet, intersection_point
        else:
            return None, None

    def goal_type(self, state, goal, route, goal_types=None):
        if goal_types is not None and len(goal_types) == 1:
            return goal_types[0]

        # get the goal type, based on the route
        goal_point = BasicPoint2d(*goal)
        path = route.shortestPath()
        start_heading = state.heading
        end_heading = LaneletHelpers.heading_at(path[-1], goal_point)
        angle_to_goal = np.diff(np.unwrap([end_heading, start_heading]))[0]

        if -np.pi / 8 < angle_to_goal < np.pi / 8:
            return 'straight-on'
        elif np.pi / 8 <= angle_to_goal < np.pi * 3 / 4:
            return 'turn-right'
        elif -np.pi / 8 >= angle_to_goal > np.pi * -3 / 4:
            return 'turn-left'
        else:
            return 'u-turn'


class GoalDetector:
    """ Detects the goals of agents based on their trajectories"""

    def __init__(self, possible_goals, dist_threshold=1.5):
        self.dist_threshold = dist_threshold
        self.possible_goals = possible_goals

    def detect_goals(self, frames):
        goals = []
        goal_frames = []
        for frame in frames:
            agent_point = np.array([frame.x, frame.y])
            for goal_idx, goal_point in enumerate(self.possible_goals):
                dist = np.linalg.norm(agent_point - goal_point)
                if dist <= self.dist_threshold and goal_idx not in goals:
                    goals.append(goal_idx)
                    goal_frames.append(frame.frame_id)
        return goals, goal_frames

    def get_agents_goals_ind(self, tracks, static_info, meta_info, map_meta, agent_class='car'):
        goal_locations = map_meta.goals
        agent_goals = {}
        for track_idx in range(len(static_info)):
            if static_info[track_idx]['class'] == agent_class:
                track = tracks[track_idx]
                agent_goals[track_idx] = []

                for i in range(static_info[track_idx]['numFrames']):
                    point = np.array([track['xCenter'][i], track['yCenter'][i]])
                    for goal_idx, loc in enumerate(goal_locations):
                        dist = np.linalg.norm(point - loc)
                        if dist < self.dist_threshold and loc not in agent_goals[track_idx]:
                            agent_goals[track_idx].append(loc)
        return agent_goals
