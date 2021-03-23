from diff_match_patch import diff_match_patch
from core.base import get_data_dir, get_base_dir

scenario_maps = {'heckstrasse': get_data_dir() + 'inD_LaneletMaps/heckstrasse.osm',
                 'bendplatz': get_data_dir() + 'inD_LaneletMaps/Bendplatz.osm',
                 'frankenberg': get_data_dir() + 'inD_LaneletMaps/frankenberg.osm',
                 'round': get_data_dir() + 'rounD-dataset/lanelets/location0.osm'}

for scenario_name, file_path in scenario_maps.items():

    text = open(file_path).read()
    dmp = diff_match_patch()
    diff = open(get_base_dir() + "/lanelet_map_patches/{}.patch".format(scenario_name)).read()
    patches = dmp.patch_fromText(diff)
    new_text, _ = dmp.patch_apply(patches, text)

    with open(get_base_dir() + '/lanelet_map_patches/{}.osm'.format(scenario_name), 'w') as f:
        f.write(new_text)
