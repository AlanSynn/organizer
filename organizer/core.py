"""
Core file organization logic.
"""

import fnmatch
import logging
import os
import shutil
import sys
import time
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Set, Any
import itertools

from .ai_service import create_ai_provider
from .config import OrganizerConfig, FolderConfig

logger = logging.getLogger(__name__)

class SpinnerProgress:
    """Progress indicator with continuous spinner animation."""

    def __init__(self, total: int, description: str = "Processing"):
        """Initialize spinner progress indicator.

        Args:
            total: Total number of items to process
            description: Description text shown before the spinner
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = 0
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spinner_index = 0
        self.batch_num = 1
        self.total_batches = 1
        self.batch_size = 0
        self.is_complete = False
        self._start_spinner_thread()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.complete()
        return False

    def _start_spinner_thread(self):
        """Start a background thread to keep the spinner moving."""
        import threading
        self._stop_spinner = False
        self._spinner_thread = threading.Thread(target=self._spin, daemon=True)
        self._spinner_thread.start()

    def _spin(self):
        """Continuously animate the spinner in a background thread."""
        while not self._stop_spinner:
            if not self.is_complete:
                current_time = time.time()
                if current_time - self.last_update_time > 0.1:
                    self.last_update_time = current_time
                    self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
                    self._display()
            time.sleep(0.1)

    def _display(self):
        """Display the spinner and batch information."""
        spinner = self.spinner_chars[self.spinner_index]

        # Calculate elapsed time and format it
        elapsed = time.time() - self.start_time
        time_str = f"{int(elapsed)}s" if elapsed < 60 else f"{int(elapsed/60)}m {int(elapsed%60)}s"

        # Format message
        if self.batch_size > 0:
            # Display batch information with spinner
            msg = f"\r{spinner} Processing batch {self.batch_num}/{self.total_batches} ({self.batch_size} items/batch) [{time_str}]"
        else:
            # Display regular progress with spinner
            msg = f"\r{spinner} {self.description} [{time_str}]"

        # Print progress
        print(msg, end='')
        sys.stdout.flush()

    def update(self, current: Optional[int] = None, description: Optional[str] = None):
        """Update the progress counter.

        Args:
            current: Current progress (increments by 1 if None)
            description: New description (keeps existing if None)
        """
        if current is not None:
            self.current = current
        else:
            self.current += 1

        if description:
            self.description = description

    def set_batch_info(self, current_batch: int, total_batches: int, batch_size: int = 0):
        """Set batch information for display.

        Args:
            current_batch: Current batch number
            total_batches: Total number of batches
            batch_size: Number of items in the current batch
        """
        self.batch_num = current_batch
        self.total_batches = total_batches
        self.batch_size = batch_size
        self.description = f"Processing batch {current_batch}/{total_batches}"

    def complete(self):
        """Mark the progress as complete."""
        self.current = self.total
        self.is_complete = True
        self._stop_spinner = True

        # Final display
        print(f"\r✓ Processing complete! Processed {self.total} items in {self.total_batches} batches")
        sys.stdout.flush()

class FileOrganizer:
    """Main file organization service."""

    def __init__(self, config: OrganizerConfig, log_level: int = logging.WARNING):
        """Initialize with configuration.

        Args:
            config: The organizer configuration
            log_level: The logging level (default: WARNING)
        """
        self.config = config
        self.ai_provider = create_ai_provider(config.ai_config)

        # Configure logger level for this module
        logger.setLevel(log_level)

    def check_outdated_folders(self, days: int = 7) -> List[FolderConfig]:
        """Check which folders haven't been organized for more than the specified days.

        Args:
            days: Number of days after which a folder is considered outdated.

        Returns:
            List of folder configs that are outdated.
        """
        outdated_folders = []

        for folder in self.config.source_folders:
            if folder.is_outdated(days):
                outdated_folders.append(folder)

        return outdated_folders

    def organize_outdated(self, days: int = 7, dry_run: bool = False, skip_organized: bool = True) -> Dict[Path, Dict[Path, Optional[Path]]]:
        """Organize files in folders that haven't been organized recently.

        Args:
            days: Number of days after which a folder is considered outdated
            dry_run: Whether to perform a dry run without making changes
            skip_organized: Whether to skip files that have already been organized

        Returns:
            Dict mapping folders to their organization results
        """
        outdated = self.check_outdated_folders(days)
        results = {}

        if not outdated:
            logger.info("No outdated folders found")
            return results

        for folder_info in outdated:
            folder = folder_info["path"]  # Use dictionary access since folder_info is a dict, not an object
            files = self.scan_folder(folder, skip_organized)

            if not files:
                logger.info(f"No files found in outdated folder: {folder}")
                results[folder] = {}
                continue

            logger.info(f"Found {len(files)} files in outdated folder: {folder}")

            # Load existing structure if available
            existing_structure = None
            if skip_organized:
                existing_structure = self.load_existing_structure()

            # Prepare file batches with content
            file_batches = []
            for file_path in files:
                # Try to read file content for context (only for text files)
                file_content = None
                if self._is_text_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            file_content = f.read(5000)  # Read first 5000 characters
                    except Exception as e:
                        logger.debug(f"Could not read file content: {e}")

                file_batches.append((file_path, file_content))

            # Categorize files in bulk for better efficiency
            categorizations = {}
            try:
                bulk_results = self.ai_provider.categorize_files_bulk(file_batches)
                for (file_path, _), categorization in zip(file_batches, bulk_results):
                    categorizations[file_path] = categorization
            except Exception as e:
                logger.warning(f"Bulk categorization failed: {e}. Falling back to individual categorization.")
                # Fall back to individual processing
                for file_path, file_content in file_batches:
                    try:
                        categorization = self.ai_provider.categorize_file(file_path, file_content)
                        categorizations[file_path] = categorization
                    except Exception as ex:
                        logger.error(f"Failed to categorize file {file_path}: {ex}")
                        categorizations[file_path] = {
                            "category_path": ["Uncategorized"],
                            "new_filename": file_path.name
                        }

            # Create folder-specific plan
            folder_plan = []
            for file_path, categorization in categorizations.items():
                target_path = self.get_target_path(file_path, categorization)
                folder_plan.append((file_path, target_path, categorization))

            # If this is a dry run, don't actually move files
            if dry_run:
                results[folder] = {source_path: None for source_path, _, _ in folder_plan}
                continue

            # Execute the plan
            folder_results = {}
            for source_path, target_path, _ in folder_plan:
                folder_results[source_path] = self.move_file(source_path, target_path)

            results[folder] = folder_results

            # Update last organized timestamp for this folder
            self._update_last_organized([folder])

            # Save the structure file with the folder plan
            self.save_structure_file(folder_plan)

        return results

    def scan_folder(self, folder: Path, skip_organized: bool = True) -> List[Path]:
        """Scan a folder for files to organize, excluding specified directories and already organized files.

        Args:
            folder: The folder to scan
            skip_organized: Whether to skip files that have already been organized

        Returns:
            List of file paths to organize
        """
        if not folder.exists() or not folder.is_dir():
            logger.warning(f"Folder does not exist or is not a directory: {folder}")
            return []

        # Load existing structure if skipping organized files
        existing_structure = None
        if skip_organized:
            existing_structure = self.load_existing_structure()

        # Get excluded directories (default to empty list if not present)
        excluded_directories = getattr(self.config, 'excluded_directories', [])
        excluded_extensions = getattr(self.config, 'excluded_extensions', [])
        exclude_patterns = getattr(self.config, 'exclude_patterns', [])

        files = []
        for item in folder.rglob("*"):
            # Skip directories, hidden files, and files in excluded directories
            if item.is_dir() or item.name.startswith("."):
                continue

            # Skip files in excluded directories
            if any(excluded in item.parts for excluded in excluded_directories):
                continue

            # Skip excluded extensions
            if item.suffix.lower() in excluded_extensions:
                continue

            # Skip files that should not be organized based on exclusion patterns
            if any(item.match(pattern) for pattern in exclude_patterns):
                continue

            # Skip already organized files if requested
            if skip_organized and existing_structure and self._is_already_organized(item, existing_structure):
                logger.debug(f"Skipping already organized file: {item}")
                continue

            files.append(item)

        return files

    def scan_all_folders(self, skip_organized: bool = True) -> List[Path]:
        """Scan all configured source folders for files to organize.

        Args:
            skip_organized: Whether to skip files that have already been organized

        Returns:
            List of file paths to organize
        """
        all_files = []
        for folder in self.config.source_folders:
            all_files.extend(self.scan_folder(folder.path, skip_organized))

        return all_files

    def _is_already_organized(self, file_path: Path, structure: Dict[str, Any]) -> bool:
        """Check if a file has already been organized based on the existing structure.

        Args:
            file_path: The file path to check
            structure: The existing organization structure

        Returns:
            True if the file has already been organized, False otherwise
        """
        # Extract filename and extension
        filename = file_path.name
        extension = file_path.suffix.lower()

        # Check if the file is in any of the examples
        for category, examples in structure.get("category_examples", {}).items():
            for example in examples:
                # If the filename matches exactly or has the same extension
                if example == filename or Path(example).suffix.lower() == extension:
                    # Check if the file with this name already exists in target location
                    category_parts = category.split("/") if "/" in category else [category]
                    target_dir = self.config.target_folder
                    for part in category_parts:
                        target_dir = target_dir / part

                    target_path = target_dir / filename
                    if target_path.exists():
                        return True

        return False

    def organize_all(self, dry_run: bool = False, incremental: bool = True, skip_organized: bool = True) -> Dict[Path, Optional[Path]]:
        """Organize all files in configured source folders.

        Args:
            dry_run: Whether to perform a dry run without making changes
            incremental: Whether to use existing structure for incremental organization
            skip_organized: Whether to skip files that have already been organized

        Returns:
            Dict mapping source paths to target paths
        """
        # If incremental is True, use existing structure if available
        if incremental:
            existing_structure = self.load_existing_structure()
        else:
            existing_structure = None

        files = self.scan_all_folders(skip_organized)

        if not files:
            logger.info("No files found to organize")
            return {}

        logger.info(f"Found {len(files)} files to organize")

        # Categorize files, using existing structure if incremental is True
        categorizations = self.categorize_files_bulk(files, existing_structure)

        # Create the plan
        plan = []
        for file_path, categorization in categorizations.items():
            target_path = self.get_target_path(file_path, categorization)
            plan.append((file_path, target_path, categorization))

        # If this is a dry run, don't actually move files
        if dry_run:
            return {source_path: None for source_path, _, _ in plan}

        # Execute the plan
        results = {}

        with SpinnerProgress(len(plan), "Moving files") as pb:
            for source_path, target_path, _ in plan:
                results[source_path] = self.move_file(source_path, target_path)
                pb.update()

        # Update last organized timestamp for each source folder
        self._update_last_organized()

        # Save the structure file
        self.save_structure_file(plan)

        return results

    def organize_folder(self, folder: FolderConfig, dry_run: bool = False) -> Dict[Path, Optional[Path]]:
        """Organize a specific folder.

        Args:
            folder: The folder configuration to organize.
            dry_run: Whether to perform a dry run without making changes.

        Returns:
            Dict mapping source paths to target paths for processed files.
        """
        results = {}
        source_path = folder.path
        files_to_process = []  # Collect files for bulk processing

        if not source_path.exists():
            logger.warning(f"Source folder does not exist: {source_path}")
            return results

        # First collect all files to process
        logger.info(f"Scanning folder {source_path} for files to process...")
        for root, _, files in os.walk(source_path, topdown=True):
            root_path = Path(root)

            # Skip if we've gone too deep
            rel_path = root_path.relative_to(source_path)
            if str(rel_path) != "." and len(rel_path.parts) > folder.max_depth:
                continue

            for filename in files:
                file_path = root_path / filename

                # Check if file matches any pattern
                if not self._matches_patterns(filename, folder.patterns):
                    continue

                # Try to read file content for context (only for text files)
                file_content = None
                if self._is_text_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            file_content = f.read(5000)  # Read first 5000 characters
                    except Exception as e:
                        logger.debug(f"Could not read file content: {e}")

                # Add to files to process
                files_to_process.append((file_path, file_content))

        # Process files in bulk
        if files_to_process:
            total_files = len(files_to_process)
            logger.info(f"Found {total_files} files to process in {source_path}")

            # Initialize progress bar
            progress = SpinnerProgress(total_files, description=f"Organizing {source_path.name}")

            try:
                # Group files in batches
                batch_size = 50
                processed_count = 0

                for i in range(0, len(files_to_process), batch_size):
                    batch_files = files_to_process[i:i+batch_size]

                    # Update progress description for current batch
                    current_batch_num = i//batch_size + 1
                    total_batches = (total_files + batch_size - 1)//batch_size
                    current_batch_size = len(batch_files)
                    progress.set_batch_info(current_batch_num, total_batches, current_batch_size)

                    # Get categorizations in bulk
                    categorizations = self.ai_provider.categorize_files_bulk(batch_files)

                    # Process each result
                    for j, ((file_path, _), categorization) in enumerate(zip(batch_files, categorizations)):
                        try:
                            # Create target path
                            target_dir = self.config.target_folder
                            for category in categorization["category_path"]:
                                target_dir = target_dir / category

                            # Ensure target directory exists
                            if not dry_run:
                                target_dir.mkdir(parents=True, exist_ok=True)

                            # Get target filename
                            new_filename = categorization["new_filename"]
                            target_path = target_dir / new_filename

                            # Move file
                            if not dry_run:
                                results[file_path] = self.move_file(source_path, target_path)
                            else:
                                logger.info(f"Would move {file_path} -> {target_path}")
                                results[file_path] = target_path
                        except Exception as e:
                            logger.error(f"Failed to process file {file_path}: {e}")
                            results[file_path] = None

                        # Update progress bar for each file
                        processed_count += 1
                        progress.update()

                # Mark progress as complete
                progress.complete()

            except Exception as e:
                logger.error(f"Failed to process files in bulk: {e}")
                # Fallback to individual processing
                fallback_progress = SpinnerProgress(len(files_to_process), "Processing files individually")
                for i, (file_path, file_content) in enumerate(files_to_process):
                    result = self._process_single_file(file_path, file_content, dry_run)
                    results[file_path] = result

                    # Update progress
                    fallback_progress.update()

                # Mark progress as complete
                fallback_progress.complete()

        return results

    def create_plan(self, skip_organized: bool = True) -> List[Tuple[Path, Path, Dict[str, Union[List[str], str]]]]:
        """Create an organization plan without executing it.

        Args:
            skip_organized: Whether to skip files that have already been organized

        Returns:
            A list of tuples (source_path, target_path, categorization_info)
        """
        plan = []

        # Load existing structure if skipping organized files
        existing_structure = None
        if skip_organized:
            existing_structure = self.load_existing_structure()

        # Get files from all source folders
        files = self.scan_all_folders(skip_organized)

        if not files:
            logger.info("No files found to organize")
            return plan

        logger.info(f"Creating plan for {len(files)} files")

        # Initialize progress bar
        progress = SpinnerProgress(len(files), description="Creating plan")
        processed_count = 0

        try:
            # Group files in batches
            batch_size = 50
            file_batches = []

            # Prepare batches with file content
            for file_path in files:
                # Try to read file content for context (only for text files)
                file_content = None
                if self._is_text_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            file_content = f.read(5000)  # Read first 5000 characters
                    except Exception as e:
                        logger.debug(f"Could not read file content: {e}")

                file_batches.append((file_path, file_content))

            # Process file batches
            for i in range(0, len(file_batches), batch_size):
                batch = file_batches[i:i+batch_size]

                # Update progress description for current batch
                current_batch_num = i//batch_size + 1
                total_batches = (len(file_batches) + batch_size - 1)//batch_size
                current_batch_size = len(batch)
                progress.set_batch_info(current_batch_num, total_batches, current_batch_size)

                # Get categorization for all files in batch
                categorizations = self.ai_provider.categorize_files_bulk(batch)

                # Process each result
                for j, ((file_path, _), categorization) in enumerate(zip(batch, categorizations)):
                    try:
                        # Create target path
                        target_dir = self.config.target_folder
                        for category in categorization["category_path"]:
                            target_dir = target_dir / category

                        # Get target filename
                        new_filename = categorization["new_filename"]
                        target_path = target_dir / new_filename

                        # Check for potential conflicts
                        if target_path.exists():
                            # We would create a unique name during actual execution
                            target_path = self._get_unique_path(target_path)

                        # Add to plan
                        plan.append((file_path, target_path, categorization))
                    except Exception as e:
                        logger.error(f"Failed to plan for file {file_path}: {e}")

                    # Update progress bar for each file
                    processed_count += 1
                    progress.update()

            # Mark progress as complete
            progress.complete()

        except Exception as e:
            logger.error(f"Failed to process files in bulk: {e}")
            # Fallback to individual processing
            fallback_progress = SpinnerProgress(len(files), "Processing files individually")
            for i, file_path in enumerate(files):
                try:
                    # Extract content from the file
                    file_content = self._extract_file_content(file_path)

                    categorization = self.ai_provider.categorize_file(file_path, file_content)

                    # Create target path
                    target_dir = self.config.target_folder
                    for category in categorization["category_path"]:
                        target_dir = target_dir / category

                    # Get target filename
                    new_filename = categorization["new_filename"]
                    target_path = target_dir / new_filename

                    # Check for potential conflicts
                    if target_path.exists():
                        target_path = self._get_unique_path(target_path)

                    # Add to plan
                    plan.append((file_path, target_path, categorization))
                except Exception as ex:
                    logger.error(f"Failed to plan for file {file_path}: {ex}")

                # Update progress
                fallback_progress.update()

            # Mark progress as complete
            fallback_progress.complete()

        return plan

    def create_plan_for_outdated(self, days: int = 7, skip_organized: bool = True) -> List[Tuple[Path, Path, Dict[str, Union[List[str], str]]]]:
        """Create a plan for outdated folders only.

        Args:
            days: Number of days after which a folder is considered outdated.
            skip_organized: Whether to skip files that have already been organized

        Returns:
            A list of tuples (source_path, target_path, categorization_info)
        """
        outdated_folders = self.check_outdated_folders(days)
        plan = []

        if not outdated_folders:
            logger.info("No outdated folders found")
            return plan

        logger.info(f"Found {len(outdated_folders)} outdated folders")

        # Process each outdated folder
        for folder_path in outdated_folders:
            # Get files in this folder
            folder_files = self.scan_folder(folder_path, skip_organized)

            if not folder_files:
                logger.info(f"No files found in outdated folder {folder_path}")
                continue

            logger.info(f"Processing {len(folder_files)} files in outdated folder {folder_path}")

            # Process each file in the folder
            for file_path in folder_files:
                try:
                    # Extract content for categorization
                    content = self._extract_file_content(file_path)

                    # Get categorization
                    categorization = self.ai_provider.categorize_file(file_path, content, self.config.get_categories())

                    # Get target path
                    target_path = self.get_target_path(file_path, categorization)

                    # Add to plan
                    plan.append((file_path, target_path, categorization))
                except Exception as e:
                    logger.warning(f"Error creating plan for outdated file {file_path}: {e}")
                    # Use default categorization for files that can't be categorized
                    categorization = {
                        "category": "Uncategorized",
                        "subcategory": "Error",
                        "reason": f"Failed to categorize: {str(e)}"
                    }
                    target_path = self.get_target_path(file_path, categorization)
                    plan.append((file_path, target_path, categorization))

            # Update last organized timestamp for this folder
            self._update_last_organized([folder_path])

        return plan

    def execute_plan(self, plan: List[Tuple[Path, Path, Dict[str, Any]]]) -> Dict[Path, Optional[Path]]:
        """Execute an organization plan and update the structure file."""
        results = {}

        if not plan:
            return results

        total = len(plan)
        progress = SpinnerProgress(total, description="Executing organization plan")

        for i, (source_path, target_path, categorization) in enumerate(plan):
            try:
                # Move file using our move_file method which handles duplicates
                actual_target_path = self.move_file(source_path, target_path)
                results[source_path] = actual_target_path
            except Exception as e:
                logger.error(f"Failed to move file {source_path}: {e}")
                results[source_path] = None

            progress.update()

        # Save structure file and log
        self.save_structure_file(plan)

        return results

    def process_file(self, file_path: Path, dry_run: bool = False) -> Optional[Path]:
        """Process a single file."""
        try:
            # Try to read file content for context (only for text files)
            file_content = None
            if self._is_text_file(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_content = f.read(5000)  # Read first 5000 characters
                except Exception as e:
                    logger.debug(f"Could not read file content: {e}")

            return self._process_single_file(file_path, file_content, dry_run)
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return None

    def _process_single_file(self, file_path: Path, file_content: Optional[str], dry_run: bool) -> Optional[Path]:
        """Internal method to process a single file."""
        try:
            # Get categorization from AI
            categorization = self.ai_provider.categorize_file(file_path, file_content)

            # Create target path
            target_dir = self.config.target_folder
            for category in categorization["category_path"]:
                target_dir = target_dir / category

            # Ensure target directory exists
            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)

            # Get target filename
            new_filename = categorization["new_filename"]
            target_path = target_dir / new_filename

            # Move file
            if not dry_run:
                return self.move_file(file_path, target_path)
            else:
                logger.info(f"Would move {file_path} -> {target_path}")
                return target_path
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return None

    def _matches_patterns(self, filename: str, patterns: List[str]) -> bool:
        """Check if filename matches any pattern."""
        if not patterns or "*" in patterns:
            return True

        return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)

    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is likely a text file based on extension."""
        text_extensions = {
            '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml',
            '.csv', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.sh', '.bat'
        }
        return file_path.suffix.lower() in text_extensions

    def _get_unique_path(self, path: Path) -> Path:
        """Get a unique path by adding a numbered suffix."""
        counter = 1
        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        while True:
            new_path = parent / f"{stem} (copy {counter}){suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def save_structure_file(self, categorized_files: List[Tuple[Path, Path, Dict[str, Any]]]) -> None:
        """Save the current organization structure to .organizer.structure file."""
        target_folder = self.config.target_folder
        structure_file = target_folder / ".organizer.structure"
        log_dir = target_folder / ".organize"

        # Create log directory if it doesn't exist
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        # Generate structure
        structure = {}
        file_examples = {}

        for source_path, target_path, categorization in categorized_files:
            # Extract relative path from target folder
            rel_path = target_path.relative_to(target_folder)
            category_path = categorization["category_path"]

            # Build the structure
            current = structure
            for category in category_path:
                if category not in current:
                    current[category] = {}
                current = current[category]

                # Store file examples (up to 3 per category)
                if category not in file_examples:
                    file_examples[category] = []

                if len(file_examples[category]) < 3:
                    file_examples[category].append(source_path.name)

        # Save structure with examples
        structure_data = {
            "directory_structure": structure,
            "category_examples": file_examples,
            "last_updated": datetime.now().isoformat(),
            "total_directories": len(set("/".join(cat["category_path"]) for _, _, cat in categorized_files)),
            "total_files": len(categorized_files)
        }

        with open(structure_file, 'w') as f:
            json.dump(structure_data, f, indent=2)

        # Save log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"organize_log_{timestamp}.json"

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "files_processed": len(categorized_files),
            "files": [
                {
                    "source": str(source),
                    "target": str(target),
                    "categories": cat["category_path"]
                }
                for source, target, cat in categorized_files
            ]
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Saved structure file to {structure_file}")
        logger.info(f"Saved log file to {log_file}")

    def load_existing_structure(self) -> Dict[str, Any]:
        """Load existing organization structure from .organizer.structure file."""
        structure_file = self.config.target_folder / ".organizer.structure"

        if not structure_file.exists():
            return {"categories": {}, "examples": {}, "descriptions": {}}

        try:
            with open(structure_file, "r", encoding="utf-8") as f:
                structure = json.load(f)
            return structure
        except Exception as e:
            logger.error(f"Failed to load existing structure: {e}")
            return {"categories": {}, "examples": {}, "descriptions": {}}

    def _merge_with_existing_structure(self, source_path: Path, file_content: Optional[str]) -> Dict[str, str]:
        """Try to categorize a file based on existing structure before using AI."""
        structure = self.load_existing_structure()

        # Check file extension
        file_ext = source_path.suffix.lower()

        # Look for matching patterns in existing structure
        for category_path, examples in structure.get("examples", {}).items():
            # Check if there are files with the same extension
            if any(Path(example).suffix.lower() == file_ext for example in examples):
                # Found a potential match by extension
                return {
                    "category_path": category_path.split(os.sep),
                    "new_filename": source_path.name
                }

        # No match found in existing structure, use AI categorization
        return self.ai_provider.categorize_file(source_path, file_content)

    def _update_last_organized(self, folders: Optional[List[Path]] = None) -> None:
        """Update the last organized timestamp for specified folders or all source folders."""
        try:
            # Determine which folders to update
            if not folders:
                # Use all source folders
                for folder in self.config.source_folders:
                    self.config.update_folder_timestamp(folder.path)
            else:
                # Update only the specified folders
                for folder_path in folders:
                    self.config.update_folder_timestamp(folder_path)

            # Save the config with updated timestamps
            self.config.save_config()
            logger.info(f"Updated last organized timestamp for {len(folders) if folders else len(self.config.source_folders)} folders")
        except Exception as e:
            logger.error(f"Error updating last organized timestamps: {e}")

    def move_file(self, source_path: Path, target_path: Path) -> Optional[Path]:
        """Move a file to the target path, handling duplicates.

        If a file with the same name and size exists in the target location,
        it will be moved to a 'duplicated items' folder in the source directory
        without renaming the file.

        Args:
            source_path: Source file path
            target_path: Target file path

        Returns:
            The actual path the file was moved to, or None if the move failed
        """
        try:
            # Create target directory
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file with same name exists in target directory
            if target_path.exists():
                # Check if they have the same size (duplicate)
                if source_path.stat().st_size == target_path.stat().st_size:
                    # Create duplicated items directory in source location
                    duplicate_dir = source_path.parent / "duplicated items"
                    duplicate_dir.mkdir(parents=True, exist_ok=True)

                    # Move to duplicated items folder with original name
                    duplicate_path = duplicate_dir / source_path.name

                    # If the file already exists in duplicated items, generate a unique name
                    # but without adding "(copy X)" suffix
                    if duplicate_path.exists():
                        # Add a timestamp to make the filename unique
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        stem = duplicate_path.stem
                        suffix = duplicate_path.suffix
                        duplicate_path = duplicate_dir / f"{stem}_{timestamp}{suffix}"

                    # Move the file to duplicated items folder
                    shutil.move(str(source_path), str(duplicate_path))
                    logger.info(f"Duplicate found: {source_path} -> {duplicate_path}")

                    return duplicate_path
                else:
                    # Not a duplicate, just a name conflict
                    # Create a unique name without "(copy X)" suffix
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    stem = target_path.stem
                    suffix = target_path.suffix
                    target_path = target_path.parent / f"{stem}_{timestamp}{suffix}"

            # Move the file
            shutil.move(str(source_path), str(target_path))
            logger.info(f"Moved {source_path} -> {target_path}")
            return target_path

        except Exception as e:
            logger.error(f"Failed to move file {source_path}: {e}")
            return None

    def get_target_path(self, file_path: Path, categorization: Dict) -> Path:
        """Get the target path for a file based on its categorization.

        Args:
            file_path: Path to the source file
            categorization: Categorization dictionary from the AI provider

        Returns:
            Target path for the file
        """
        # Start with the target folder
        target_dir = self.config.target_folder

        # Add category and subcategory to path
        if "category" in categorization:
            target_dir = target_dir / categorization["category"]

            # Add subcategory if present
            if "subcategory" in categorization and categorization["subcategory"]:
                target_dir = target_dir / categorization["subcategory"]

        # If we have a category_path list instead (old format)
        elif "category_path" in categorization:
            for category in categorization["category_path"]:
                target_dir = target_dir / category

        # Use the original filename or new_filename if provided
        if "new_filename" in categorization and categorization["new_filename"]:
            filename = categorization["new_filename"]
        else:
            filename = file_path.name

        # Create the full target path
        target_path = target_dir / filename

        # Check for file conflicts and create a unique path if needed
        if target_path.exists():
            target_path = self._get_unique_path(target_path)

        return target_path