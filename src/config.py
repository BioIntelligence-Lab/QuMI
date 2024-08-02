import yaml


def read_config_file(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_config(path, kwargs):
    cfg = read_config_file(path)
    # CLI args override config for now. Do not specify hyperparameters in the config
    cfg = kwargs | cfg
    return cfg


if __name__ == '__main__':
    print(read_config_file('config.yml'))
