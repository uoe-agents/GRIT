import matplotlib.pyplot as plt
from lanelet2.core import LaneletMap, Lanelet, getId, LineString3d, Point3d, AttributeMap
from core.lanelet_helpers import LaneletHelpers


def tuple_list_to_ls(tuple_list):
    return LineString3d(getId(), [Point3d(getId(), x, y) for x, y in tuple_list])


def get_test_map():
    """
    Map with this Layout:

    3             ___
                 /5 /
    2    _______/__/
        |  2  |   /4|
    1   |_____|__/__|
        |  1  |  3  |
    0   |_____|_____|

        0  1  2  3  4
    """

    ll_1_right_start = Point3d(getId(), 0, 0)
    ll_1_left_start = Point3d(getId(), 0, 1)
    ll_1_right_end = Point3d(getId(), 2, 0)
    ll_1_left_end = Point3d(getId(), 2, 1)
    ll_2_left_start = Point3d(getId(), 0, 2)
    ll_2_left_end = Point3d(getId(), 2, 2)
    ll_3_right_end = Point3d(getId(), 4, 0)
    ll_3_left_end = Point3d(getId(), 4, 1)
    ll_4_left_end = Point3d(getId(), 4, 2)
    ll_5_left_end = Point3d(getId(), 3, 3)
    ll_5_right_end = Point3d(getId(), 4, 3)

    ls_attribute_map_dashed = AttributeMap({'subtype': 'dashed',
                                            'type': 'line_thin'})

    ll_1_right_bound = LineString3d(getId(), [ll_1_right_start, ll_1_right_end], ls_attribute_map_dashed)
    ll_1_left_bound = LineString3d(getId(), [ll_1_left_start, ll_1_left_end], ls_attribute_map_dashed)
    ll_2_left_bound = LineString3d(getId(), [ll_2_left_start, ll_2_left_end], ls_attribute_map_dashed)
    ll_3_right_bound = LineString3d(getId(), [ll_1_right_end, ll_3_right_end], ls_attribute_map_dashed)
    ll_3_left_bound = LineString3d(getId(), [ll_1_left_end, ll_3_left_end], ls_attribute_map_dashed)
    ll_4_left_bound = LineString3d(getId(), [ll_2_left_end, ll_4_left_end], ls_attribute_map_dashed)
    ll_5_left_bound = LineString3d(getId(), [ll_2_left_end, ll_5_left_end], ls_attribute_map_dashed)
    ll_5_right_bound = LineString3d(getId(), [ll_1_left_end, ll_5_right_end], ls_attribute_map_dashed)

    ll_attribute_map = AttributeMap({'location': 'urban',
                                     'one_way': 'yes',
                                     'region': 'de',
                                     'subtype': 'road',
                                     'type': 'lanelet'})

    ll_1 = Lanelet(1, ll_1_left_bound, ll_1_right_bound, ll_attribute_map)
    ll_2 = Lanelet(2, ll_2_left_bound, ll_1_left_bound, ll_attribute_map)
    ll_3 = Lanelet(3, ll_3_left_bound, ll_3_right_bound, ll_attribute_map)
    ll_4 = Lanelet(4, ll_4_left_bound, ll_3_left_bound, ll_attribute_map)
    ll_5 = Lanelet(5, ll_5_left_bound, ll_5_right_bound, ll_attribute_map)

    lanelet_map = LaneletMap()
    lanelet_map.add(ll_1)
    lanelet_map.add(ll_2)
    lanelet_map.add(ll_3)
    lanelet_map.add(ll_4)
    lanelet_map.add(ll_5)

    return lanelet_map


def plot_test_map():
    test_map = get_test_map()
    for ll in test_map.laneletLayer:
        LaneletHelpers.plot(ll)
    plt.show()


def get_test_lanelet_straight():
    left_bound = tuple_list_to_ls([(0, 1), (1, 1)])
    right_bound = tuple_list_to_ls([(0, 0), (1, 0)])
    lanelet = Lanelet(getId(), left_bound, right_bound)
    return lanelet


def get_following_lanelets():
    left_points = [Point3d(getId(), x, y) for x, y in [(0, 1), (1, 1), (2, 2)]]
    right_points = [Point3d(getId(), x, y) for x, y in [(0, 0), (1, 0), (2, 1)]]
    lanelet_1_left_bound = LineString3d(getId(), left_points[:2])
    lanelet_1_right_bound = LineString3d(getId(), right_points[:2])
    lanelet_1 = Lanelet(getId(), lanelet_1_left_bound, lanelet_1_right_bound)
    lanelet_2_left_bound = LineString3d(getId(), left_points[1:])
    lanelet_2_right_bound = LineString3d(getId(), right_points[1:])
    lanelet_2 = Lanelet(getId(), lanelet_2_left_bound, lanelet_2_right_bound)
    return lanelet_1, lanelet_2


def get_adjacent_lanelets():
    leftmost_bound = tuple_list_to_ls([(0, 2), (1, 2), (2, 2)])
    mid_bound = tuple_list_to_ls([(0, 1), (1, 1), (2, 2)])
    rightmost_bound = tuple_list_to_ls([(0, 0), (1, 0), (2, 1)])
    left_lanelet = Lanelet(getId(), leftmost_bound, mid_bound)
    right_lanelet = Lanelet(getId(), mid_bound, rightmost_bound)
    return left_lanelet, right_lanelet


def get_test_lanelet_curved():
    left_bound = tuple_list_to_ls([(0, 1), (1, 1), (2, 2)])
    right_bound = tuple_list_to_ls([(0, 0), (1, 0), (2, 1)])
    lanelet = Lanelet(getId(), left_bound, right_bound)
    return lanelet