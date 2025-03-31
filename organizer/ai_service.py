"""
AI service for file categorization using LLM package.

References:
- https://llm.datasette.io/en/stable/index.html
"""

import json
import logging
import os
import re
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Any, Tuple

import llm

from .config import AIConfig
from .utils import parse_ai_response, sanitize_filename

logger = logging.getLogger(__name__)

# Default templates for file categorization
DEFAULT_CATEGORIZATION_TEMPLATE = """
You are a file organization assistant. Your task is to analyze files and suggest appropriate
categorization for better organization. For each file, you should:
1. Analyze the filename and optional content
2. Suggest appropriate category folders (max 3 levels deep)
3. Suggest a clear, descriptive filename in the same language as the original filename that maintains the original extension.

IMPORTANT: Never use 'Uncategorized' unless you absolutely cannot determine a better category.
Even with limited information, try to assign a meaningful category based on:
- File extension (e.g., .pdf → Documents, .jpg → Images)
- Filename patterns (e.g., IMG_1234 → Photos, tax_2023 → Finance/Taxes)
- Content preview if available
- Language indicators in the filename

Respond in the following JSON format only:
{
    "category_path": ["level1", "level2", "level3"],
    "new_filename": "descriptive_name.ext"
}
"""

class AIProvider(Protocol):
    """Protocol for AI providers."""
    def categorize_file(self, file_path: Path, file_content: Optional[str] = None) -> Dict[str, str]:
        """Categorize a file and suggest organization details."""
        ...

    def categorize_files_bulk(self, files: List[Tuple[Path, Optional[str]]]) -> List[Dict[str, str]]:
        """Categorize multiple files in bulk."""
        ...

class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(self, max_requests_per_minute: int = 30):
        self.max_requests_per_minute = max_requests_per_minute
        self.request_timestamps = []

    def wait_if_needed(self):
        """Wait if rate limit is reached."""
        current_time = time.time()

        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps
                                   if current_time - ts < 60]

        # Check if we've reached the limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            # Get the oldest timestamp in the last minute
            oldest = min(self.request_timestamps)
            # Calculate how long we need to wait
            wait_time = 60 - (current_time - oldest)
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)

        # Record this request
        self.request_timestamps.append(time.time())

class LLMProvider:
    """LLM package-based provider for AI services."""

    def __init__(self, config: AIConfig):
        """Initialize LLM provider with configuration.

        Args:
            config: AI configuration including model type, API keys, etc.
        """
        self.config = config

        # Set up model ID based on configuration
        if config.model_type.lower() == "openai":
            self.model_id = config.model_name or "gpt-4o-mini"
        else:
            # Use model_name as model_id for other providers
            self.model_id = config.model_name or "default"

        # Try to get the model, will raise exception if not available
        try:
            self.model = llm.get_model(self.model_id)
            logger.info(f"Using LLM model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise

        # Initialize template management
        self._init_templates()

        # Initialize rate limiter
        self.rate_limiter = RateLimiter()

    def _init_templates(self):
        """Initialize templates for the provider."""
        # Template name for file categorization
        self.template_name = "file_categorization"

        # Path to store templates
        self.template_dir = Path(llm.user_dir()) / "templates"
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.template_path = self.template_dir / f"{self.template_name}.yaml"

        # Check if template already exists
        if not self.template_path.exists():
            # Create default template
            template_data = {
                "name": self.template_name,
                "description": "Template for file categorization",
                "system": DEFAULT_CATEGORIZATION_TEMPLATE,
                "prompt": "{{prompt}}"
            }

            # Save template
            with open(self.template_path, "w") as f:
                yaml.dump(template_data, f)

            logger.info(f"Created default template: {self.template_name}")

    def get_template(self) -> str:
        """Get the current template content."""
        try:
            if self.template_path.exists():
                with open(self.template_path, "r") as f:
                    template_data = yaml.safe_load(f)
                    return template_data.get("system", DEFAULT_CATEGORIZATION_TEMPLATE)
            return DEFAULT_CATEGORIZATION_TEMPLATE
        except Exception as e:
            logger.warning(f"Failed to get template {self.template_name}: {e}")
            return DEFAULT_CATEGORIZATION_TEMPLATE

    def update_template(self, new_content: str) -> bool:
        """Update the template with new content.

        Args:
            new_content: New template content

        Returns:
            bool: Success status
        """
        try:
            # Load existing template data or create new
            template_data = {}
            if self.template_path.exists():
                with open(self.template_path, "r") as f:
                    template_data = yaml.safe_load(f) or {}

            # Update system prompt
            template_data.update({
                "name": self.template_name,
                "description": "Template for file categorization",
                "system": new_content,
                "prompt": "{{prompt}}"
            })

            # Save updated template
            with open(self.template_path, "w") as f:
                yaml.dump(template_data, f)

            logger.info(f"Updated template: {self.template_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update template: {e}")
            return False

    def categorize_file(self, file_path: Path, file_content: Optional[str] = None) -> Dict[str, str]:
        """Categorize a file using LLM."""
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()

        # Prepare prompt with file info
        prompt = f"File to organize: {file_path.name}"
        if file_content:
            prompt += f"\nContent preview: {file_content[:1000]}"

        try:
            # Get model options based on configuration
            model_options = {}
            if self.config.model_params:
                model_options.update(self.config.model_params)

            # Set API key if available
            if self.config.api_key:
                model_options["api_key"] = self.config.api_key

            # Use the template for system prompt
            system_prompt = self.get_template()

            # Execute the model
            response = self.model.prompt(
                prompt,
                system=system_prompt,
                **model_options
            )

            # Parse response
            result = parse_ai_response(str(response))

            # Sanitize filename
            result["new_filename"] = sanitize_filename(result["new_filename"])

            # Ensure extension is preserved
            original_ext = file_path.suffix
            if not result["new_filename"].endswith(original_ext):
                result["new_filename"] = f"{Path(result['new_filename']).stem}{original_ext}"

            return result

        except Exception as e:
            logger.error(f"Error with LLM API: {e}")
            return {
                "category_path": ["Uncategorized"],
                "new_filename": file_path.name
            }

    def categorize_files_bulk(self, files: List[Tuple[Path, Optional[str]]]) -> List[Dict[str, str]]:
        """Categorize multiple files in a single API call if possible, or sequentially."""
        if not files:
            return []

        # Group files in batches of 10 to avoid overloading the API
        batch_size = 10
        all_results = []

        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]

            # Try batch processing first
            try:
                batch_results = self._process_batch(batch_files)
                all_results.extend(batch_results)
            except Exception as e:
                logger.warning(f"Batch processing failed, falling back to individual processing: {e}")
                # Process each file individually
                for file_path, file_content in batch_files:
                    result = self.categorize_file(file_path, file_content)
                    all_results.append(result)

        return all_results

    def _process_batch(self, files: List[Tuple[Path, Optional[str]]]) -> List[Dict[str, str]]:
        """Process a batch of files with a single API call if supported by model."""
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()

        # Construct a prompt for all files in the batch
        file_prompts = []
        for idx, (file_path, file_content) in enumerate(files):
            file_prompt = f"File {idx+1}: {file_path.name}"
            if file_content:
                file_prompt += f"\nContent preview: {file_content[:500]}"
            file_prompts.append(file_prompt)

        batch_prompt = "\n\n".join(file_prompts)
        batch_prompt += "\n\nCategorize all files above in order. Respond with a JSON array containing categorization for each file. For each file, provide meaningful categories, never use Uncategorized unless absolutely necessary."

        try:
            # Get model options based on configuration
            model_options = {}
            if self.config.model_params:
                model_options.update(self.config.model_params)

            # Set API key if available
            if self.config.api_key:
                model_options["api_key"] = self.config.api_key

            # Use the template for system prompt
            system_prompt = self.get_template()

            # Execute the model
            response = self.model.prompt(
                batch_prompt,
                system=system_prompt,
                **model_options
            )

            # Try to extract JSON array from response
            try:
                json_pattern = r'\[.*\]'
                match = re.search(json_pattern, str(response), re.DOTALL)
                if match:
                    json_array = json.loads(match.group(0))

                    # Process each result
                    results = []
                    for idx, result_dict in enumerate(json_array):
                        if idx < len(files):
                            file_path = files[idx][0]
                            result = parse_ai_response(json.dumps(result_dict))

                            # Sanitize filename
                            result["new_filename"] = sanitize_filename(result["new_filename"])

                            # Ensure extension is preserved
                            original_ext = file_path.suffix
                            if not result["new_filename"].endswith(original_ext):
                                result["new_filename"] = f"{Path(result['new_filename']).stem}{original_ext}"

                            results.append(result)

                    # If we got fewer results than files, add default values for the rest
                    while len(results) < len(files):
                        file_path = files[len(results)][0]
                        results.append({
                            "category_path": ["Uncategorized"],
                            "new_filename": file_path.name
                        })

                    return results
            except Exception as e:
                logger.error(f"Failed to parse bulk response: {e}")
                raise

        except Exception as e:
            logger.error(f"Error with LLM API bulk request: {e}")
            raise

def create_ai_provider(config: AIConfig) -> AIProvider:
    """Create an AI provider based on configuration."""
    return LLMProvider(config)