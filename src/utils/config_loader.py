import yaml
import os

def get_reliability_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'config', 'config.yml'
    )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('reliability', {})

def get_full_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'config', 'config.yml'
    )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}