"""
Basic tests for the organizer package.
"""

import json
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, ANY

import pytest

from organizer.config import AIConfig, FolderConfig, OrganizerConfig
from organizer.core import FileOrganizer
from organizer.utils import parse_ai_response, sanitize_filename
from organizer.ai_service import LLMProvider, DEFAULT_CATEGORIZATION_TEMPLATE


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
    assert sanitize_filename('file:with:colons.txt') == 'file_with_colons.txt'
    assert sanitize_filename('file*with?invalid"chars.txt') == 'file_with_invalid_chars.txt'

    # Test length limit
    long_name = 'a' * 300 + '.txt'
    assert len(sanitize_filename(long_name)) <= 255


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


@pytest.fixture
def mock_llm_model():
    """Mock the LLM model."""
    model = MagicMock()
    model.prompt.return_value = '{"category_path": ["Documents", "Reports"], "new_filename": "report_2023.pdf"}'
    return model


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


def test_llm_provider_init(mock_llm_model):
    """Test LLMProvider initialization and template management."""
    # Mock the user_dir call to return a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        template_dir = Path(tmpdir) / "templates"
        template_dir.mkdir()
        template_path = template_dir / "file_categorization.yaml"

        with patch('llm.get_model', return_value=mock_llm_model), \
             patch('llm.user_dir', return_value=tmpdir), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir', return_value=None), \
             patch('builtins.open', mock_open()) as m:

            # Create config for LLM provider
            config = AIConfig(
                model_type="llm",
                api_key="fake_key",
                model_name="gpt-4o-mini"
            )

            # Initialize provider
            provider = LLMProvider(config)

            # Check initialization
            assert provider.template_dir == template_dir
            assert provider.template_path == template_path

            # Verify template file was opened for writing
            m.assert_called_with(template_path, 'w')

            # Verify something was written (not checking exact content)
            handle = m()
            assert handle.write.call_count > 0


def test_llm_provider_get_template(mock_llm_model):
    """Test LLMProvider get_template method."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('llm.get_model', return_value=mock_llm_model), \
             patch('llm.user_dir', return_value=tmpdir), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.open', mock_open(read_data=yaml.dump({
                 'system': 'Test template content'
             }))):

            # Create config for LLM provider
            config = AIConfig(
                model_type="llm",
                api_key="fake_key",
                model_name="gpt-4o-mini"
            )

            # Initialize provider
            provider = LLMProvider(config)

            # Test get_template
            template_content = provider.get_template()
            assert template_content == 'Test template content'


def test_llm_provider_update_template(mock_llm_model):
    """Test LLMProvider update_template method."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('llm.get_model', return_value=mock_llm_model), \
             patch('llm.user_dir', return_value=tmpdir), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.open', mock_open(read_data=yaml.dump({
                 'system': 'Initial template content'
             }))) as m:

            # Create config for LLM provider
            config = AIConfig(
                model_type="llm",
                api_key="fake_key",
                model_name="gpt-4o-mini"
            )

            # Initialize provider
            provider = LLMProvider(config)

            # Reset mock to track new calls
            m.reset_mock()

            # Test update_template
            new_template = "Updated template content"
            result = provider.update_template(new_template)

            # Check result
            assert result is True

            # Verify file was opened for reading and writing
            assert m.call_count >= 2

            # Check that something was written
            handle = m()
            assert handle.write.call_count > 0


def test_llm_provider_categorize_file(mock_llm_model):
    """Test LLMProvider categorize_file method."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('llm.get_model', return_value=mock_llm_model), \
             patch('llm.user_dir', return_value=tmpdir), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.open', mock_open(read_data=yaml.dump({
                 'system': DEFAULT_CATEGORIZATION_TEMPLATE
             }))):

            # Create config for LLM provider
            config = AIConfig(
                model_type="llm",
                api_key="fake_key",
                model_name="gpt-4o-mini"
            )

            # Initialize provider
            provider = LLMProvider(config)

            # Test categorize_file
            file_path = Path("test.pdf")
            result = provider.categorize_file(file_path, "Sample content")

            # Check results
            assert result["category_path"] == ["Documents", "Reports"]
            assert result["new_filename"] == "report_2023.pdf"

            # Verify model was called correctly
            mock_llm_model.prompt.assert_called_once()
            args, kwargs = mock_llm_model.prompt.call_args
            assert "File to organize: test.pdf" in args[0]
            assert "Sample content" in args[0]
            assert kwargs.get("system") == DEFAULT_CATEGORIZATION_TEMPLATE