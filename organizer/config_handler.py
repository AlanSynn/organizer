"""
Configuration file handling for the organizer CLI tool.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import AIConfig, FolderConfig, OrganizerConfig

logger = logging.getLogger(__name__)

# ISO format for datetime serialization
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse config file: {e}")
        raise ValueError(f"Invalid configuration file format: {e}")
    except Exception as e:
        logger.error(f"Failed to read config file: {e}")
        raise FileNotFoundError(f"Could not read configuration file: {e}")

def save_config_file(config_path: Path, config_data: Dict[str, Any]) -> None:
    """Save configuration to file."""
    try:
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config file: {e}")
        raise IOError(f"Could not save configuration file: {e}")

def create_default_config(
    target_folder: Path,
    model_type: str,
    api_key: Optional[str] = None,
    model_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model_name: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None
) -> OrganizerConfig:
    """Create a default configuration."""
    ai_config = AIConfig(
        model_type=model_type,
        api_key=api_key,
        model_path=model_path,
        api_base=api_base,
        model_name=model_name,
        model_params=model_params
    )

    return OrganizerConfig(
        source_folders=[],
        target_folder=target_folder,
        ai_config=ai_config
    )

def add_source_folder(
    config: OrganizerConfig,
    folder_path: Path,
    patterns: list[str],
    max_depth: int
) -> OrganizerConfig:
    """Add a source folder to the configuration."""
    # Check if folder already exists
    for folder in config.source_folders:
        if folder.path == folder_path:
            # Update existing folder
            folder.patterns = patterns
            folder.max_depth = max_depth
            return config

    # Add new folder
    folder_config = FolderConfig(
        path=folder_path,
        max_depth=max_depth,
        patterns=patterns,
        last_organized=None  # Never organized
    )

    config.source_folders.append(folder_config)
    return config

def remove_source_folder(
    config: OrganizerConfig,
    folder_path: Path
) -> OrganizerConfig:
    """Remove a source folder from the configuration."""
    config.source_folders = [
        folder for folder in config.source_folders
        if folder.path != folder_path
    ]
    return config

def config_to_dict(config: OrganizerConfig) -> Dict[str, Any]:
    """Convert config to dictionary for serialization."""
    result = {
        "target_folder": str(config.target_folder),
        "source_folders": [],
        "ai_config": config.ai_config.to_dict()
    }

    # Convert source folders
    for folder in config.source_folders:
        folder_dict = folder.dict(exclude_none=True)
        folder_dict["path"] = str(folder.path)

        # Convert datetime to string if present
        if folder.last_organized:
            folder_dict["last_organized"] = folder.last_organized.strftime(DATETIME_FORMAT)

        result["source_folders"].append(folder_dict)

    return result

def dict_to_config(config_dict: Dict[str, Any]) -> OrganizerConfig:
    """Convert dictionary to config object."""
    # Convert AI config using the from_dict method
    ai_config = AIConfig.from_dict(config_dict.get("ai_config", {}))

    # Convert source folders
    source_folders = []
    for folder_dict in config_dict.get("source_folders", []):
        # Convert string path to Path
        folder_dict["path"] = Path(folder_dict["path"])

        # Convert string datetime to datetime object if present
        if "last_organized" in folder_dict and folder_dict["last_organized"]:
            try:
                folder_dict["last_organized"] = datetime.strptime(
                    folder_dict["last_organized"],
                    DATETIME_FORMAT
                )
            except ValueError:
                logger.warning(f"Invalid datetime format for last_organized: {folder_dict['last_organized']}")
                folder_dict["last_organized"] = None

        source_folders.append(FolderConfig(**folder_dict))

    return OrganizerConfig(
        source_folders=source_folders,
        target_folder=Path(config_dict["target_folder"]),
        ai_config=ai_config
    )