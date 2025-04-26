"""
Configuration management for the Typeface Killer pipeline.
Loads configuration from YAML files and sets up default paths.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and set up default paths.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Dictionary containing configuration settings with default paths added
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Set default paths relative to the project root
    project_root = Path(__file__).parent.parent.parent
    config["paths"] = {
        "data": str(project_root / "data"),
        "models": str(project_root / "data" / "models"),
        "output": str(project_root / "data" / "output")
    }
    
    return config 