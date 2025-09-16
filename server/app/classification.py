import os
import json
import yaml
import sys
from PIL import Image

# Update system path to include custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import custom modules
from utils.prediction.predict_multitask import predict
from utils.load_config import load_config


def get_classification(images: str, config: dict) -> list:
    """Get prediction from AI model.

    Args:
        images (str): Path to the images.
        config (dict): Configuration dictionary.

    Returns:
        list: Result [task, diseases].
    """
    prediction = predict(image_path=images, config=config)
    print(type(config))
    return prediction
