"""
Configuration file handling for the organizer CLI tool.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .config import OrganizerConfig, FolderConfig, AIConfig

def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

def save_config_file(config_path: Path, config_dict: Dict[str, Any]) -> None:
    """Save configuration to JSON file."""
    # Create parent directories if they don't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=json_serializer)

def json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def dict_to_config(config_dict: Dict[str, Any]) -> OrganizerConfig:
    """Convert dictionary to OrganizerConfig object."""
    # Convert source folders
    source_folders = []
    for folder_dict in config_dict.get("source_folders", []):
        # Convert path to Path object
        folder_dict["path"] = Path(folder_dict["path"])

        # Convert datetime string to datetime object
        if folder_dict.get("last_organized"):
            try:
                folder_dict["last_organized"] = datetime.fromisoformat(
                    folder_dict["last_organized"]
                )
            except ValueError:
                folder_dict["last_organized"] = None

        source_folders.append(FolderConfig(**folder_dict))

    # Convert target folder to Path object
    target_folder = Path(config_dict["target_folder"])

    # Convert AI config
    ai_config_dict = config_dict.get("ai_config", {})

    # Handle model_path if it exists
    if ai_config_dict.get("model_path"):
        ai_config_dict["model_path"] = Path(ai_config_dict["model_path"])

    # Handle LLM-specific configs
    if ai_config_dict.get("model_type", "").lower() == "llm":
        # Ensure model_params has appropriate defaults for LLM
        if "model_params" not in ai_config_dict:
            ai_config_dict["model_params"] = {}

    ai_config = AIConfig.from_dict(ai_config_dict)

    # Create OrganizerConfig object
    return OrganizerConfig(
        source_folders=source_folders,
        target_folder=target_folder,
        ai_config=ai_config,
    )

def config_to_dict(config: OrganizerConfig) -> Dict[str, Any]:
    """Convert OrganizerConfig object to dictionary."""
    config_dict = {
        "source_folders": [],
        "target_folder": str(config.target_folder),
        "ai_config": config.ai_config.to_dict(),
    }

    for folder in config.source_folders:
        folder_dict = {
            "path": str(folder.path),
            "max_depth": folder.max_depth,
            "patterns": folder.patterns,
        }

        if folder.last_organized:
            folder_dict["last_organized"] = folder.last_organized.isoformat()

        config_dict["source_folders"].append(folder_dict)

    return config_dict

def create_default_config(target_folder: Path, api_key: Optional[str] = None, model_type: str = "llm") -> OrganizerConfig:
    """Create default configuration."""
    # Create target folder if it doesn't exist
    target_folder.mkdir(parents=True, exist_ok=True)

    # Create default AI config
    if model_type.lower() == "llm":
        # Default configuration for LLM package
        default_model = "gpt-4o-mini"  # Use a reasonable default

        ai_config = AIConfig(
            model_type="llm",
            api_key=api_key,
            model_name=default_model,
            model_params={
                "temperature": 0.1,
            }
        )
    elif model_type.lower() == "gemini":
        # Legacy Gemini configuration for backward compatibility
        api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        ai_config = AIConfig(
            model_type="gemini",
            api_key=api_key,
            model_name="gemini-2.0-flash-lite",
            model_params={
                "temperature": 0.1,
            }
        )
    elif model_type.lower() == "llama":
        # Legacy Llama configuration for backward compatibility
        model_path = os.environ.get("LLAMA_MODEL_PATH", "")
        model_path = Path(model_path) if model_path else None

        ai_config = AIConfig(
            model_type="llama",
            model_path=model_path,
            model_params={
                "n_ctx": 4096,
                "n_threads": 4,
                "temperature": 0.1,
            }
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create default organizer config
    return OrganizerConfig(
        source_folders=[],
        target_folder=target_folder,
        ai_config=ai_config,
    )

def add_source_folder(config: OrganizerConfig, folder_path: Path, max_depth: int = 2, patterns: Optional[List[str]] = None) -> OrganizerConfig:
    """Add a source folder to configuration."""
    # Check if folder exists
    if not folder_path.exists():
        raise FileNotFoundError(f"Source folder not found at {folder_path}")

    # Check if folder is already in config
    for existing_folder in config.source_folders:
        if existing_folder.path == folder_path:
            raise ValueError(f"Folder {folder_path} is already in config")

    # Create folder config
    folder_config = FolderConfig(
        path=folder_path,
        max_depth=max_depth,
        patterns=patterns or [],
    )

    # Add to source folders
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