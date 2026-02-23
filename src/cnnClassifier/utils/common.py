import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import joblib
import json
from box import ConfigBox
from ensure import ensure_annotations
from pathlib import path
from typing import Any
import base64



@ensure_annotations
def read_yaml (path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns
    
    Args:
        path_to_yaml (str): path like input
        
    Raises:
        ValueError: if yaml file is empty
        e: empty file
        
        
    Returns:
        ConfigBox: ConfigBox type
        
    """
    
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError ("yaml file is empty")
    except Exception as e:
        raise e
    
    
@ensure_annotations
def create_directories (path_to_directories: list, verbose=True):
    """create list of directories
    
    
    Args:
    path_to_directories (list): list of path of directories
    ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False 
    """
        
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbse:
            logger.info(f"created directory at: {path}")
            
            

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data
    
    
    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file  
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        
    logger.info(f"json file saved at: {path}")