import pytest
import numpy as np
from lanelet2.core import BasicPoint2d
from core.lanelet_helpers import LaneletHelpers
from test.lanelet_test_helpers import get_test_lanelet_straight, get_test_lanelet_curved, get_following_lanelets, \
    get_adjacent_lanelets

from test.lanelet_test_helpers import get_test_map


def test_dist_along():
    lanelet = get_test_lanelet_straight()
    point = BasicPoint2d(0.5, 0.75)
    assert LaneletHelpers.dist_along(lanelet, point) == pytest.approx(0.5)


def test_heading_at_center_straight():
    lanelet = get_test_lanelet_straight()
    point = BasicPoint2d(0.5, 0.75)
    assert LaneletHelpers.heading_at(lanelet, point) == pytest.approx(0)


def test_heading_at_center_start():
    lanelet = get_test_lanelet_straight()
    point = BasicPoint2d(0., 0.75)
    assert LaneletHelpers.heading_at(lanelet, point) == pytest.approx(0)


def test_heading_at_center_end():
    lanelet = get_test_lanelet_straight()
    point = BasicPoint2d(1., 0.75)
    assert LaneletHelpers.heading_at(lanelet, point) == pytest.approx(0)


def test_heading_at_center_curved():
    lanelet = get_test_lanelet_curved()
    point = BasicPoint2d(1.5, 0.75)
    assert LaneletHelpers.heading_at(lanelet, point) == pytest.approx(np.pi/4)


def test_following_lanelets_true():
    lanelet_1, lanelet_2 = get_following_lanelets()
    assert LaneletHelpers.follows(lanelet_2, lanelet_1)


def test_following_lanelets_false():
    lanelet_1, lanelet_2 = get_following_lanelets()
    assert not LaneletHelpers.follows(lanelet_1, lanelet_2)


def test_left_of_true():
    left_lanelet, right_lanelet = get_adjacent_lanelets()
    assert LaneletHelpers.left_of(left_lanelet, right_lanelet)


def test_left_of_false():
    left_lanelet, right_lanelet = get_adjacent_lanelets()
    assert not LaneletHelpers.left_of(right_lanelet, left_lanelet)


def test_overlap_area_no_overlap():
    map = get_test_map()
    l1 = map.laneletLayer.get(1)
    l2 = map.laneletLayer.get(2)
    l4 = map.laneletLayer.get(4)
    overlap_area, overlap_centroid = LaneletHelpers.overlap_area(l1, l2)
    assert overlap_area == 0
    overlap_area, overlap_centroid = LaneletHelpers.overlap_area(l1, l4)
    assert overlap_area == 0


def test_overlap_area_overlap():
    map = get_test_map()
    l4 = map.laneletLayer.get(4)
    l5 = map.laneletLayer.get(5)
    overlap_area, overlap_centroid = LaneletHelpers.overlap_area(l4, l5)
    assert overlap_area == 0.5
    assert overlap_centroid.x == 7/3
    assert overlap_centroid.y == 5/3
