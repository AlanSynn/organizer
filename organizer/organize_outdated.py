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