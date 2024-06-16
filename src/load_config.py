import yaml

def load_config(config_path='src/config.yaml'):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
if __name__ == "__main__":
    config = load_config()
    print(config)
