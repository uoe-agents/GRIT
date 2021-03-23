from collections import Counter

import numpy as np
import pytest
from lanelet2.core import BasicPoint2d
from core.scenario import AgentState, Frame
from core.feature_extraction import FeatureExtractor
from test.lanelet_test_helpers import get_test_lanelet_straight, get_test_lanelet_curved, get_test_map


def get_feature_extractor():
    lanelet_map = get_test_map()
    return FeatureExtractor(lanelet_map)


def test_angle_in_lane_straight():
    state = AgentState(0, 0.5, 0.75, 0, 0, np.pi/4, 0, 0, 0, 0, 0, 0)
    lanelet = get_test_lanelet_straight()
    assert FeatureExtractor.angle_in_lane(state, lanelet) == pytest.approx(np.pi/4)


def test_angle_in_lane_curved():
    state = AgentState(0, 1.5, 1.0, 0, 0, np.pi/2, 0, 0, 0, 0, 0, 0)
    lanelet = get_test_lanelet_curved()
    assert FeatureExtractor.angle_in_lane(state, lanelet) == pytest.approx(np.pi/4)


def test_route_to_goal_straight():
    feature_extractor = get_feature_extractor()
    start_lanelet = feature_extractor.lanelet_map.laneletLayer.get(1)
    goal = (3.5, 0.5)
    route = feature_extractor.route_to_goal(start_lanelet, goal)
    assert route is not None
    assert [l.id for l in route.shortestPath()] == [1, 3]


def test_route_to_goal_turn():
    feature_extractor = get_feature_extractor()
    start_lanelet = feature_extractor.lanelet_map.laneletLayer.get(1)
    goal = (3.0, 2.5)
    route = feature_extractor.route_to_goal(start_lanelet, goal)
    assert route is not None
    assert [l.id for l in route.shortestPath()] == [1, 2, 5]


def get_current_lanelet_simple():
    feature_extractor = get_feature_extractor()
    state = AgentState(0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    lanelet = feature_extractor.get_current_lanelet(state)
    assert lanelet.id == 1


def get_current_lanelet_straight():
    feature_extractor = get_feature_extractor()
    state = AgentState(0, 3, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    lanelet = feature_extractor.get_current_lanelet(state)
    assert lanelet.id == 4


def get_current_lanelet_turn():
    feature_extractor = get_feature_extractor()
    state = AgentState(0, 3, 1.5, 0, 0, np.pi/4, 0, 0, 0, 0, 0, 0)
    lanelet = feature_extractor.get_current_lanelet(state)
    assert lanelet.id == 5


def test_can_pass():
    feature_extractor = get_feature_extractor()
    lanelet_1 = feature_extractor.lanelet_map.laneletLayer.get(1)
    lanelet_2 = feature_extractor.lanelet_map.laneletLayer.get(2)
    lanelet_3 = feature_extractor.lanelet_map.laneletLayer.get(3)
    assert feature_extractor.can_pass(lanelet_1, lanelet_3)
    assert not feature_extractor.can_pass(lanelet_3, lanelet_1)
    assert feature_extractor.can_pass(lanelet_1, lanelet_2)
    assert feature_extractor.can_pass(lanelet_2, lanelet_1)
    assert not feature_extractor.can_pass(lanelet_2, lanelet_3)


def test_lanelets_at_single():
    feature_extractor = get_feature_extractor()
    lanelets = feature_extractor.lanelets_at(BasicPoint2d(1.0, 0.5))
    assert [l.id for l in lanelets] == [1]


def test_lanelets_at_multiple():
    feature_extractor = get_feature_extractor()
    lanelets = feature_extractor.lanelets_at(BasicPoint2d(2.2, 1.5))
    assert Counter([l.id for l in lanelets]) == Counter([4, 5])


def test_lanelets_at_none():
    feature_extractor = get_feature_extractor()
    lanelets = feature_extractor.lanelets_at(BasicPoint2d(1.0, 3.0))
    assert len(lanelets) == 0


def test_in_correct_lane():
    feature_extractor = get_feature_extractor()
    start_lanelet = feature_extractor.lanelet_map.laneletLayer.get(1)
    goal = (3.5, 0.5)
    route = feature_extractor.route_to_goal(start_lanelet, goal)
    assert feature_extractor.in_correct_lane(route)


def test_not_in_correct_lane():
    feature_extractor = get_feature_extractor()
    start_lanelet = feature_extractor.lanelet_map.laneletLayer.get(1)
    goal = (3.0, 2.5)
    route = feature_extractor.route_to_goal(start_lanelet, goal)
    assert not feature_extractor.in_correct_lane(route)


def test_path_to_goal_length_same_lanelet():
    feature_extractor = get_feature_extractor()
    state = AgentState(0, 2.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    goal = (3.5, 0.5)
    start_lanelet = feature_extractor.lanelet_map.laneletLayer.get(3)
    end_lanelet = start_lanelet
    route = feature_extractor.routing_graph.getRoute(start_lanelet, end_lanelet)
    assert feature_extractor.path_to_goal_length(state, goal, route) == 1


def test_path_to_goal_length_different_lanelet():
    feature_extractor = get_feature_extractor()
    state = AgentState(0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    goal = (3.5, 0.5)
    start_lanelet = feature_extractor.lanelet_map.laneletLayer.get(1)
    end_lanelet = feature_extractor.lanelet_map.laneletLayer.get(3)
    route = feature_extractor.routing_graph.getRoute(start_lanelet, end_lanelet)
    assert feature_extractor.path_to_goal_length(state, goal, route) == 3


def test_angle_to_goal_zero():
    state = AgentState(0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    goal = (3.5, 0.5)
    assert FeatureExtractor.angle_to_goal(state, goal) == 0


def test_angle_to_goal_angled():
    state = AgentState(0, 2.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    goal = (3.5, 0.5)
    assert FeatureExtractor.angle_to_goal(state, goal) == pytest.approx(np.pi/4)


def test_get_vehicles_in_front():
    feature_extractor = get_feature_extractor()
    start_lanelet = feature_extractor.lanelet_map.laneletLayer.get(1)
    goal = (3.5, 0.5)
    route = feature_extractor.route_to_goal(start_lanelet, goal)

    frame = Frame(0)
    frame.add_agent_state(0, AgentState(0, 2.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    frame.add_agent_state(1, AgentState(0, 3.0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    frame.add_agent_state(2, AgentState(0, 3.0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    vehicles = FeatureExtractor.get_vehicles_in_front(route, frame)
    assert set(vehicles) == {0, 1}


def test_vehicle_in_front():
    feature_extractor = get_feature_extractor()
    start_lanelet = feature_extractor.lanelet_map.laneletLayer.get(1)
    goal = (3.5, 0.5)
    route = feature_extractor.route_to_goal(start_lanelet, goal)

    frame = Frame(0)

    frame.add_agent_state(0, AgentState(0, 3.0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    frame.add_agent_state(1, AgentState(0, 2.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    frame.add_agent_state(2, AgentState(0, 3.0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    state = AgentState(0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    agent_id, dist = FeatureExtractor.vehicle_in_front(state, route, frame)
    assert agent_id == 1
    assert dist == 2


def test_goal_type_straight_on():
    feature_extractor = get_feature_extractor()
    start_lanelet = feature_extractor.lanelet_map.laneletLayer.get(1)
    goal = (3.5, 0.5)
    route = feature_extractor.route_to_goal(start_lanelet, goal)
    state = AgentState(0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert feature_extractor.goal_type(state, goal, route) == 'straight-on'


def test_goal_type_turn_left():
    feature_extractor = get_feature_extractor()
    start_lanelet = feature_extractor.lanelet_map.laneletLayer.get(1)
    goal = (3.0, 2.5)
    route = feature_extractor.route_to_goal(start_lanelet, goal)
    state = AgentState(0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert feature_extractor.goal_type(state, goal, route) == 'turn-left'


def test_get_goals_current_lanelets():
    feature_extractor = get_feature_extractor()
    goals = [(3.5, 0.5)]
    state = AgentState(0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    routes = feature_extractor.get_goal_routes(state, goals)
    current_lanelets = [r.shortestPath()[0] for r in routes]
    assert [l.id for l in current_lanelets] == [1]


def test_get_goals_current_lanelets_multiple():
    feature_extractor = get_feature_extractor()
    goals = [(3.5, 1.5), (3.0, 2.5)]
    state = AgentState(0, 2.2, 1.5, 0, 0, 0.1, 0, 0, 0, 0, 0, 0)
    routes = feature_extractor.get_goal_routes(state, goals)
    current_lanelets = [r.shortestPath()[0] for r in routes]
    assert [l.id for l in current_lanelets] == [4, 5]
