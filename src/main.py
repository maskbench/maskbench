import os
import yaml

from video_chunker import VideoChunker


def main():
    config = load_config()
    print(config)


def load_config():
    config_file_name = os.getenv("MASKBENCH_CONFIG_FILE", "maskbench-config.yml")
    config_file_path = os.path.join("/config", config_file_name)
    
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuration file is empty or not found.")
    
    return config


if __name__ == "__main__":
    main()