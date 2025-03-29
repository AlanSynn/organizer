"""
Utility functions for the organizer CLI tool.
"""

import json
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON object from text that might contain extra content."""
    # Try to find JSON pattern with regex
    json_pattern = r'({[^{}]*(?:{[^{}]*}[^{}]*)*})'
    match = re.search(json_pattern, text)

    if match:
        json_text = match.group(1)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning(f"Found JSON-like text but failed to parse: {json_text}")

    # Try the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not extract JSON from AI response")
        return None

def parse_ai_response(response_text: str) -> Dict[str, Union[List[str], str]]:
    """Parse AI response into a standardized format."""
    # Extract JSON from text
    json_data = extract_json_from_text(response_text)

    if not json_data:
        # Fallback to default categorization
        return {
            "category_path": ["Uncategorized"],
            "new_filename": "unknown_file"
        }

    # Validate and standardize the response
    result = {}

    # Handle category path
    if "category_path" in json_data:
        if isinstance(json_data["category_path"], list):
            # Filter out empty strings and limit to 3 levels
            category_path = [c for c in json_data["category_path"] if c and isinstance(c, str)][:3]
            result["category_path"] = category_path if category_path else ["Uncategorized"]
        else:
            # Try to handle string path
            if isinstance(json_data["category_path"], str):
                parts = [p.strip() for p in json_data["category_path"].split("/")]
                result["category_path"] = [p for p in parts if p][:3] or ["Uncategorized"]
            else:
                result["category_path"] = ["Uncategorized"]
    else:
        result["category_path"] = ["Uncategorized"]

    # Handle filename
    if "new_filename" in json_data and isinstance(json_data["new_filename"], str):
        result["new_filename"] = json_data["new_filename"]
    else:
        result["new_filename"] = "unknown_file"

    return result

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to ensure it's valid across platforms."""
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Replace control characters
    filename = ''.join(c if c.isprintable() else '_' for c in filename)

    # Limit length
    max_length = 255  # Safe limit for most file systems
    if len(filename) > max_length:
        name, ext = Path(filename).stem, Path(filename).suffix
        name = name[:max_length - len(ext) - 1]
        filename = f"{name}{ext}"

    return filename

def get_file_hash(file_path: Path, block_size: int = 65536) -> str:
    """
    Generate a hash for a file to check for duplicates or changes.

    Args:
        file_path: Path to the file
        block_size: Size of blocks to read for hashing

    Returns:
        String representation of the file's SHA-256 hash
    """
    if not file_path.exists():
        return ""

    hash_obj = hashlib.sha256()

    try:
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                hash_obj.update(block)
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash for {file_path}: {e}")
        return ""