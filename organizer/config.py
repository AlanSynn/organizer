"""
Configuration management for the organizer CLI tool.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field

class AIConfig:
    """AI model configuration."""

    def __init__(self,
                 model_type: str = "gemini",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 model_name: Optional[str] = None,
                 model_path: Optional[Path] = None,
                 model_params: Optional[Dict[str, Any]] = None):
        """Initialize AI config.

        Args:
            model_type: Type of AI model ('gemini' or 'llama')
            api_key: API key for cloud models
            api_base: Base URL for API (optional)
            model_name: Specific model name (e.g., 'gemini-2.0-flash' or 'gemini-2.0-pro')
            model_path: Path to local model file (required for llama)
            model_params: Additional model parameters
        """
        self.model_type = model_type
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.model_path = model_path
        self.model_params = model_params or {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AIConfig':
        """Create config from dictionary."""
        model_path = data.get('model_path')
        if model_path:
            model_path = Path(model_path)

        return cls(
            model_type=data.get('model_type', 'gemini'),
            api_key=data.get('api_key'),
            api_base=data.get('api_base'),
            model_name=data.get('model_name'),
            model_path=model_path,
            model_params=data.get('model_params')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'model_type': self.model_type,
            'api_key': self.api_key,
            'api_base': self.api_base,
            'model_name': self.model_name,
            'model_params': self.model_params,
        }

        if self.model_path:
            result['model_path'] = str(self.model_path)

        return result

class FolderConfig(BaseModel):
    """Configuration for a target folder."""
    path: Path = Field(..., description="Path to the target folder")
    max_depth: int = Field(3, description="Maximum folder hierarchy depth")
    patterns: List[str] = Field(default_factory=list, description="File patterns to match")
    last_organized: Optional[datetime] = Field(None, description="When this folder was last organized")

    def is_outdated(self, days: int = 7) -> bool:
        """Check if folder hasn't been organized for the specified number of days."""
        if not self.last_organized:
            return True  # Never organized

        threshold = datetime.now() - timedelta(days=days)
        return self.last_organized < threshold

class OrganizerConfig(BaseModel):
    """Main configuration for the organizer tool."""
    source_folders: List[FolderConfig] = Field(
        default_factory=list,
        description="List of folders to monitor"
    )
    target_folder: Path = Field(..., description="Base folder for organized files")
    ai_config: AIConfig = Field(..., description="AI model configuration")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """Initialize OrganizerConfig with defaults."""
        # Handle AIConfig specially if it's passed as a dict
        ai_config_data = data.get('ai_config', {})
        if isinstance(ai_config_data, dict):
            data['ai_config'] = AIConfig.from_dict(ai_config_data)

        # Initialize model with default Gemini version if none specified
        if isinstance(data['ai_config'], AIConfig) and data['ai_config'].model_type == 'gemini' and not data['ai_config'].model_name:
            data['ai_config'].model_name = 'gemini-2.0-flash'  # Using the most recent model version

        super().__init__(**data)

    @classmethod
    def load_config(cls, config_path: Optional[Path] = None) -> "OrganizerConfig":
        """Load configuration from file."""
        from .config_handler import load_config_file, dict_to_config

        # Use provided path or default to ~/.organizer.rc
        if config_path is None:
            config_path = Path.home() / ".organizer.rc"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}. "
                "Please run 'organizer init' to create one."
            )

        # Load and convert configuration
        config_dict = load_config_file(config_path)
        return dict_to_config(config_dict)

    def save_config(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        from .config_handler import save_config_file, config_to_dict

        # Use provided path or default to ~/.organizer.rc
        if config_path is None:
            config_path = Path.home() / ".organizer.rc"

        config_dict = config_to_dict(self)
        save_config_file(config_path, config_dict)

    def update_folder_timestamp(self, folder_path: Path) -> None:
        """Update the last_organized timestamp for a specific folder."""
        for folder in self.source_folders:
            if folder.path == folder_path:
                folder.last_organized = datetime.now()
                break