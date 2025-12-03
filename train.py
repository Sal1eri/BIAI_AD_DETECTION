import os
import yaml
from utils.metric import get_classification_metrics




def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(config):
    """Placeholder function for model training logic."""
    print("Training model with the following configuration:")
    print(config)
    # Add training logic here




if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_name = 'vgg16'
    config_path = os.path.join(BASE_DIR, f'configs/training/{model_name}.yaml')
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    print("Configuration Loaded:")
    print(config)
    train_model(config)



