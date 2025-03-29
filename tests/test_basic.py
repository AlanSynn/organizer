"""
Basic tests for the organizer package.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from organizer.config import AIConfig, FolderConfig, OrganizerConfig
from organizer.core import FileOrganizer
from organizer.utils import parse_ai_response, sanitize_filename


def test_parse_ai_response():
    """Test parsing AI response."""
    # Test valid JSON response
    valid_json = '{"category_path": ["Documents", "Work", "Reports"], "new_filename": "quarterly_report.pdf"}'
    result = parse_ai_response(valid_json)
    assert result["category_path"] == ["Documents", "Work", "Reports"]
    assert result["new_filename"] == "quarterly_report.pdf"

    # Test response with extra text
    messy_response = 'Here is my suggestion: {"category_path": ["Photos", "Vacation"], "new_filename": "beach_sunset.jpg"}'
    result = parse_ai_response(messy_response)
    assert result["category_path"] == ["Photos", "Vacation"]
    assert result["new_filename"] == "beach_sunset.jpg"

    # Test invalid response
    invalid_response = "I'm not sure how to categorize this file."
    result = parse_ai_response(invalid_response)
    assert result["category_path"] == ["Uncategorized"]
    assert result["new_filename"] == "unknown_file"


def test_sanitize_filename():
    """Test filename sanitization."""
    # Test invalid characters
    assert sanitize_filename('file/with:invalid*chars?.txt') == 'file_with_invalid_chars_.txt'

    # Test very long filename
    long_name = 'a' * 300 + '.txt'
    result = sanitize_filename(long_name)
    assert len(result) <= 255
    assert result.endswith('.txt')


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return OrganizerConfig(
        source_folders=[
            FolderConfig(
                path=Path("/test/source"),
                patterns=["*.txt", "*.pdf"],
                max_depth=2
            )
        ],
        target_folder=Path("/test/target"),
        ai_config=AIConfig(
            model_type="gemini",
            api_key="fake_key"
        )
    )


@pytest.fixture
def mock_ai_provider():
    """Create a mock AI provider for testing."""
    provider = MagicMock()
    provider.categorize_file.return_value = {
        "category_path": ["Documents", "Reports"],
        "new_filename": "report_2023.pdf"
    }
    return provider


def test_file_organizer(mock_config, mock_ai_provider):
    """Test file organizer core functionality."""
    with patch('organizer.core.create_ai_provider', return_value=mock_ai_provider):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup test environment
            source_dir = Path(tmpdir) / "source"
            target_dir = Path(tmpdir) / "target"
            source_dir.mkdir()
            target_dir.mkdir()

            # Update config to use temporary directories
            mock_config.source_folders[0].path = source_dir
            mock_config.target_folder = target_dir

            # Create test file
            test_file = source_dir / "test.pdf"
            with open(test_file, "w") as f:
                f.write("Test content")

            # Create organizer and process file
            organizer = FileOrganizer(mock_config)
            result = organizer.process_file(test_file)

            # Check results
            assert result is not None
            assert result == target_dir / "Documents" / "Reports" / "report_2023.pdf"
            assert result.exists()
            assert not test_file.exists()  # File should be moved