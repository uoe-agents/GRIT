import pathlib


def get_base_dir():
    return str(pathlib.Path(__file__).parent.parent.absolute())


def get_data_dir():
    return get_base_dir() + '/data/'


def get_img_dir():
    return get_base_dir() + '/images/'


def get_scenario_config_dir():
    return get_base_dir() + '/scenario_config/'


def get_dt_config_dir():
    return get_base_dir() + '/dt_config/'
