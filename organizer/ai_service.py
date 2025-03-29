"""
AI service for file categorization.

Note: This module requires the `google-genai` package (not google-generativeai).
To install: `pip install google-genai`
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Any, Tuple

from google import genai
from google.genai import types
from llama_cpp import Llama

from .config import AIConfig
from .prompts import GEMINI_CATEGORIZATION_PROMPT, LLAMA_CATEGORIZATION_PROMPT
from .utils import parse_ai_response, sanitize_filename

logger = logging.getLogger(__name__)

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

class GeminiProvider:
    """Google Gemini AI provider."""

    def __init__(self, config: AIConfig):
        if not config.api_key:
            raise ValueError("API key is required for Gemini")

        # Initialize client with API key
        self.client = genai.Client(api_key=config.api_key)

        # Set API base if provided
        if config.api_base:
            logger.info(f"Using custom API base: {config.api_base}")
            # Note: Requires manual setting in environment variables or other mechanisms
            # as the current client doesn't have a direct way to set the API base

        # Get model configuration
        self.model_name = config.model_name or "gemini-2.0-flash-lite"

        # Model generation parameters
        self.generation_config = types.GenerateContentConfig(
            temperature=0.1,
            system_instruction=GEMINI_CATEGORIZATION_PROMPT
        )

        # Update generation config with custom model params if provided
        if config.model_params:
            for param, value in config.model_params.items():
                if hasattr(self.generation_config, param):
                    setattr(self.generation_config, param, value)

        # Initialize rate limiter
        self.rate_limiter = RateLimiter()

    def categorize_file(self, file_path: Path, file_content: Optional[str] = None) -> Dict[str, str]:
        """Categorize a file using Gemini."""
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()

        prompt = f"File to organize: {file_path.name}"
        if file_content:
            prompt += f"\nContent preview: {file_content[:1000]}"

        try:
            # Send the prompt as a simple string
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.generation_config
            )

            # Parse response
            result = parse_ai_response(response.text)

            # Sanitize filename
            result["new_filename"] = sanitize_filename(result["new_filename"])

            # Ensure extension is preserved
            original_ext = file_path.suffix
            if not result["new_filename"].endswith(original_ext):
                result["new_filename"] = f"{Path(result['new_filename']).stem}{original_ext}"

            return result

        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            return {
                "category_path": ["Uncategorized"],
                "new_filename": file_path.name
            }

    def categorize_files_bulk(self, files: List[Tuple[Path, Optional[str]]]) -> List[Dict[str, str]]:
        """Categorize multiple files in a single API call."""
        if not files:
            return []

        # Group files in batches of 50 to avoid overloading the API
        batch_size = 50
        all_results = []

        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            batch_results = self._process_batch(batch_files)
            all_results.extend(batch_results)

        return all_results

    def _process_batch(self, files: List[Tuple[Path, Optional[str]]]) -> List[Dict[str, str]]:
        """Process a batch of files with a single API call."""
        # Apply rate limiting before API call
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
            # Send the batch prompt as a simple string
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=batch_prompt,
                config=self.generation_config
            )

            # Try to extract JSON array from response
            try:
                json_pattern = r'\[.*\]'
                match = re.search(json_pattern, response.text, re.DOTALL)
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

            # Fallback to individual processing
            return [self.categorize_file(file_path, file_content) for file_path, file_content in files]

        except Exception as e:
            logger.error(f"Error with Gemini API bulk request: {e}")
            # Return default categorization for all files
            return [
                {
                    "category_path": ["Uncategorized"],
                    "new_filename": file_path.name
                }
                for file_path, _ in files
            ]

class LlamaProvider:
    """Local Llama model provider."""

    def __init__(self, config: AIConfig):
        if not config.model_path:
            raise ValueError("Model path is required for Llama")

        # Basic model parameters
        model_params = {
            "model_path": str(config.model_path),
            "n_ctx": 2048,
            "n_threads": 4
        }

        # Add any custom model parameters
        if config.model_params:
            model_params.update(config.model_params)

        self.model = Llama(**model_params)

        # Initialize rate limiter for consistency with API provider
        self.rate_limiter = RateLimiter()

    def categorize_file(self, file_path: Path, file_content: Optional[str] = None) -> Dict[str, str]:
        """Categorize a file using local Llama model."""
        # Apply rate limiting for consistency
        self.rate_limiter.wait_if_needed()

        user_prompt = f"<|user|>\nFile: {file_path.name}"
        if file_content:
            user_prompt += f"\nContent preview: {file_content[:500]}"
        user_prompt += "</|user|>"

        try:
            response = self.model(
                f"{LLAMA_CATEGORIZATION_PROMPT}\n{user_prompt}\n<|assistant|>",
                max_tokens=512,
                temperature=0.7
            )

            # Parse response
            result = parse_ai_response(response["choices"][0]["text"])

            # Sanitize filename
            result["new_filename"] = sanitize_filename(result["new_filename"])

            # Ensure extension is preserved
            original_ext = file_path.suffix
            if not result["new_filename"].endswith(original_ext):
                result["new_filename"] = f"{Path(result['new_filename']).stem}{original_ext}"

            return result

        except Exception as e:
            logger.error(f"Failed to get Llama response: {e}")
            return {
                "category_path": ["Uncategorized"],
                "new_filename": file_path.name
            }

    def categorize_files_bulk(self, files: List[Tuple[Path, Optional[str]]]) -> List[Dict[str, str]]:
        """Process multiple files in batch for local Llama model.

        Note: For local models, we process sequentially but in a single method call for consistency.
        This could be optimized with parallel inference if supported by the model.
        """
        # Process files in batches of 50 for consistency with API provider
        batch_size = 50
        all_results = []

        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]

            # Process each file in the batch
            results = []
            for file_path, file_content in batch_files:
                try:
                    # Apply rate limiting
                    self.rate_limiter.wait_if_needed()

                    result = self.categorize_file(file_path, file_content)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process file {file_path} in bulk: {e}")
                    results.append({
                        "category_path": ["Uncategorized"],
                        "new_filename": file_path.name
                    })

            all_results.extend(results)

        return all_results

def create_ai_provider(config: AIConfig) -> AIProvider:
    """Create an AI provider based on configuration."""
    if config.model_type.lower() == "gemini":
        return GeminiProvider(config)
    elif config.model_type.lower() == "llama":
        return LlamaProvider(config)
    else:
        raise ValueError(f"Unsupported AI model type: {config.model_type}")