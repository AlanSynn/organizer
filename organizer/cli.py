"""
Command-line interface for the organizer tool.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.prompt import Confirm

from .config import OrganizerConfig
from .config_handler import create_default_config, add_source_folder
from .core import FileOrganizer

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Default to WARNING to hide INFO logs
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("organizer")
console = Console()

app = typer.Typer(
    name="organizer",
    help="AI-powered file organization tool",
    add_completion=True
)

# Global option for config path
CONFIG_PATH_OPTION = typer.Option(
    None,
    "--config",
    "-c",
    help="Custom path to config file (default: ~/.organizer.rc)"
)

# Global option for log level
def log_level_callback(level: str) -> int:
    """Convert string log level to corresponding logging level constant."""
    try:
        return getattr(logging, level.upper())
    except AttributeError:
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        raise typer.BadParameter(f"Log level must be one of: {', '.join(valid_levels)}")

LOG_LEVEL_OPTION = typer.Option(
    "warning",
    "--log-level",
    "-l",
    help="Set logging level (debug, info, warning, error, critical)",
    callback=log_level_callback
)

def configure_logger(log_level: int):
    """Configure logger with the specified level."""
    logger.setLevel(log_level)
    # Also set level for root logger and all its handlers
    logging.getLogger().setLevel(log_level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level)

@app.command()
def init(
    target_folder: Path = typer.Option(
        ...,
        "--target",
        "-t",
        help="Base folder for organized files"
    ),
    model_type: str = typer.Option(
        "gemini",
        "--model",
        "-m",
        help="AI model to use (gemini/llama)"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key for cloud models"
    ),
    model_path: Optional[Path] = typer.Option(
        None,
        "--model-path",
        help="Path to local model file"
    ),
    api_base: Optional[str] = typer.Option(
        None,
        "--api-base",
        help="Base URL for API calls"
    ),
    model_name: Optional[str] = typer.Option(
        None,
        "--model-name",
        help="Specific model name to use (e.g., gemini-2.0-flash-lite, gemini-pro-vision)"
    ),
    model_params_json: Optional[str] = typer.Option(
        None,
        "--model-params",
        help="Additional model parameters as JSON string"
    ),
    config_path: Optional[Path] = CONFIG_PATH_OPTION,
    log_level: str = LOG_LEVEL_OPTION
):
    """Initialize the organizer configuration."""
    # Configure logger
    configure_logger(log_level)

    try:
        # Validate input
        if model_type.lower() == "gemini" and not api_key:
            console.print("[red]Error: API key is required for Gemini model[/red]")
            raise typer.Exit(1)
        if model_type.lower() == "llama" and not model_path:
            console.print("[red]Error: Model path is required for Llama model[/red]")
            raise typer.Exit(1)

        # Parse model parameters if provided
        model_params = None
        if model_params_json:
            try:
                model_params = json.loads(model_params_json)
            except json.JSONDecodeError:
                console.print("[red]Error: Invalid JSON for model parameters[/red]")
                raise typer.Exit(1)

        # Create and save config
        config = create_default_config(
            target_folder=target_folder,
            model_type=model_type,
            api_key=api_key,
            model_path=model_path,
            api_base=api_base,
            model_name=model_name,
            model_params=model_params
        )
        config.save_config(config_path)

        actual_config_path = config_path or Path.home() / ".organizer.rc"
        console.print(f"[green]Configuration initialized successfully![/green]")
        console.print(f"Config path: [blue]{actual_config_path}[/blue]")
        console.print(f"Target folder: [blue]{target_folder}[/blue]")
        console.print(f"AI model: [blue]{model_type}{' (' + model_name + ')' if model_name else ''}[/blue]")
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        raise typer.Exit(1)

@app.command()
def add_source(
    folder: Path = typer.Argument(
        ...,
        help="Folder to monitor",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    patterns: List[str] = typer.Option(
        ["*"],
        "--pattern",
        "-p",
        help="File patterns to match (e.g., '*.pdf')"
    ),
    max_depth: int = typer.Option(
        3,
        "--max-depth",
        "-d",
        help="Maximum folder hierarchy depth"
    ),
    config_path: Optional[Path] = CONFIG_PATH_OPTION,
    log_level: str = LOG_LEVEL_OPTION
):
    """Add a source folder to monitor."""
    # Configure logger
    configure_logger(log_level)

    try:
        # Load existing config
        config = OrganizerConfig.load_config(config_path)

        # Add source folder
        config = add_source_folder(
            config=config,
            folder_path=folder,
            patterns=patterns,
            max_depth=max_depth
        )

        # Save updated config
        config.save_config(config_path)

        console.print(f"[green]Added source folder: {folder}[/green]")
        console.print(f"File patterns: [blue]{', '.join(patterns)}[/blue]")
        console.print(f"Maximum depth: [blue]{max_depth}[/blue]")
    except Exception as e:
        logger.error(f"Failed to add source folder: {e}")
        raise typer.Exit(1)

@app.command()
def organize(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without making changes"
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        "-p",
        help="Create a plan without executing it"
    ),
    auto_confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Automatically confirm and execute the plan without asking"
    ),
    check_outdated: bool = typer.Option(
        False,
        "--outdated-only",
        "-o",
        help="Only organize folders that haven't been organized in more than the specified days"
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--no-incremental",
        help="Use existing directory structure for incremental organization"
    ),
    current_dir: bool = typer.Option(
        False,
        "--now",
        "-n",
        help="Organize files within the current directory (source and target are the same)"
    ),
    skip_organized: bool = typer.Option(
        True,
        "--skip-organized/--include-organized",
        help="Skip files that have already been organized (default: True)"
    ),
    days: int = typer.Option(
        7,
        "--days",
        "-d",
        help="Number of days after which a folder is considered outdated"
    ),
    config_path: Optional[Path] = CONFIG_PATH_OPTION,
    log_level: str = LOG_LEVEL_OPTION
):
    """Organize files in source folders with planning and confirmation."""
    # Configure logger
    configure_logger(log_level)

    try:
        # Load configuration
        config = OrganizerConfig.load_config(config_path)

        # If organizing within current directory, temporarily set target to current dir
        original_target = None
        if current_dir:
            original_target = config.target_folder
            current_path = Path.cwd()
            config.target_folder = current_path

            # Also add current directory as a source folder if not already present
            current_dir_in_sources = False
            for folder in config.source_folders:
                if folder.path == current_path:
                    current_dir_in_sources = True
                    break

            if not current_dir_in_sources:
                from .config_handler import add_source_folder
                config = add_source_folder(
                    config=config,
                    folder_path=current_path,
                    patterns=["*"],
                    max_depth=3
                )

            console.print(f"[bold blue]Organizing files within current directory: [/bold blue]{current_path}")

        # Create organizer
        organizer = FileOrganizer(config)

        # Check for existing structure file
        structure_file = config.target_folder / ".organizer.structure"
        has_existing_structure = structure_file.exists()

        if incremental and has_existing_structure:
            console.print(f"[blue]Found existing structure file. Will use it for incremental organization.[/blue]")
        elif incremental and not has_existing_structure:
            console.print(f"[yellow]No existing structure file found. Will create one after organization.[/yellow]")
        elif not incremental:
            console.print(f"[yellow]Incremental mode disabled. Will use AI for all categorization decisions.[/yellow]")

        if skip_organized:
            console.print(f"[blue]Skipping files that have already been organized.[/blue]")
        else:
            console.print(f"[yellow]Including all files, even those that have already been organized.[/yellow]")

        if check_outdated:
            outdated_folders = organizer.check_outdated_folders(days)

            if not outdated_folders:
                console.print(f"[green]No folders need organizing (all organized within the last {days} days)[/green]")
                # Restore original target if we were using current directory
                if current_dir and original_target:
                    config.target_folder = original_target
                return

            console.print(f"[yellow]Found {len(outdated_folders)} folders that haven't been organized in {days} days:[/yellow]")
            for folder in outdated_folders:
                last_organized = folder.last_organized.strftime("%Y-%m-%d %H:%M:%S") if folder.last_organized else "Never"
                console.print(f"  • {folder.path} (Last organized: {last_organized})")

        if plan_only or not auto_confirm:
            # First create a plan
            console.print("[yellow]Creating organization plan...[/yellow]")

            if check_outdated:
                plan = organizer.create_plan_for_outdated(days, skip_organized=skip_organized)
            else:
                plan = organizer.create_plan(skip_organized=skip_organized)

            if not plan:
                console.print("[yellow]No files found to organize[/yellow]")
                # Restore original target if we were using current directory
                if current_dir and original_target:
                    config.target_folder = original_target
                return

            # Display plan to user
            console.print(f"\n[bold]Organization Plan ({len(plan)} files):[/bold]")

            # Display source and target base directories
            common_source_base = None
            if all(source_path.parts[:1] == plan[0][0].parts[:1] for source_path, _, _ in plan):
                common_source_base = Path(plan[0][0].parts[0])
                for i in range(1, len(plan[0][0].parts)):
                    potential_base = Path(*plan[0][0].parts[:i+1])
                    if all(str(source_path).startswith(str(potential_base)) for source_path, _, _ in plan):
                        common_source_base = potential_base
                    else:
                        break

            target_base = config.target_folder

            # Quote paths that contain whitespace for better display
            source_display = f'"{common_source_base}"' if common_source_base and ' ' in str(common_source_base) else common_source_base
            target_display = f'"{target_base}"' if ' ' in str(target_base) else target_base

            console.print(f"[bold blue]Source directory:[/bold blue] {source_display or 'Multiple source directories'}")
            console.print(f"[bold green]Target directory:[/bold green] {target_display}\n")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Source", style="dim", overflow="fold")
            table.add_column("Target", style="green", overflow="fold")
            table.add_column("Categories", style="cyan", overflow="fold")

            for source_path, target_path, categorization in plan:
                # Use relative paths if we have a common source base
                if common_source_base and str(source_path).startswith(str(common_source_base)):
                    try:
                        rel_source = source_path.relative_to(common_source_base)
                        source_display = f".../{rel_source}"
                    except ValueError:
                        source_display = f'"{source_path}"' if ' ' in str(source_path) else str(source_path)
                else:
                    source_display = f'"{source_path}"' if ' ' in str(source_path) else str(source_path)

                # Always use relative path for target since we know the base
                rel_target = target_path.relative_to(target_base)
                target_display = f".../{rel_target}"

                # Quote paths with whitespace in table display
                if ' ' in source_display and not source_display.startswith('"'):
                    source_display = f'"{source_display}"'
                if ' ' in target_display and not target_display.startswith('"'):
                    target_display = f'"{target_display}"'

                categories = " > ".join(categorization["category_path"])
                table.add_row(
                    source_display,
                    target_display,
                    categories
                )

            console.print(table)

            if plan_only:
                # Restore original target if we were using current directory
                if current_dir and original_target:
                    config.target_folder = original_target
                return

            # Ask for confirmation
            if not auto_confirm:
                confirmed = Confirm.ask("\nDo you want to execute this plan?")
                if not confirmed:
                    console.print("[yellow]Operation cancelled by user[/yellow]")
                    # Restore original target if we were using current directory
                    if current_dir and original_target:
                        config.target_folder = original_target
                    return

        # If we're doing a dry run or if the plan was confirmed
        if dry_run:
            console.print("[yellow]Running in dry-run mode (no changes will be made)[/yellow]")
            if check_outdated:
                results = organizer.organize_outdated(days, dry_run=True, skip_organized=skip_organized)
                processed = sum(len(folder_results) for folder_results in results.values())
            else:
                results = organizer.organize_all(dry_run=True, incremental=incremental, skip_organized=skip_organized)
                processed = len(results)

            console.print(f"[yellow]Would process {processed} files[/yellow]")
        elif auto_confirm or (not plan_only and not auto_confirm):
            # Either auto-confirmed or manually confirmed
            if not auto_confirm:
                console.print("[green]Executing organization plan...[/green]")

            if check_outdated:
                if 'plan' in locals():
                    # Use the plan we created
                    results = organizer.execute_plan(plan)
                    processed = len(results)
                    moved = sum(1 for result in results.values() if result is not None)
                else:
                    # Organize outdated folders directly
                    results = organizer.organize_outdated(days, dry_run=False, skip_organized=skip_organized)
                    processed = sum(len(folder_results) for folder_results in results.values())
                    moved = sum(sum(1 for result in folder_results.values() if result is not None)
                               for folder_results in results.values())
            else:
                if 'plan' in locals():
                    # Use the plan we created
                    results = organizer.execute_plan(plan)
                else:
                    # Create and execute plan in one step
                    results = organizer.organize_all(dry_run=False, incremental=incremental, skip_organized=skip_organized)

                processed = len(results)
                moved = sum(1 for result in results.values() if result is not None)

            console.print(f"[green]Processed {processed} files, moved {moved}[/green]")

            # Show structure file and log location
            structure_file = config.target_folder / ".organizer.structure"
            log_dir = config.target_folder / ".organize"

            if structure_file.exists():
                console.print(f"[blue]Updated organization structure: [/blue]{structure_file}")
            if log_dir.exists():
                console.print(f"[blue]Organization logs directory: [/blue]{log_dir}")

        # Restore original target if we were using current directory
        if current_dir and original_target:
            config.target_folder = original_target

    except Exception as e:
        logger.error(f"Failed to organize files: {e}")
        # Restore original target if we were using current directory
        if current_dir and original_target:
            config.target_folder = original_target
        raise typer.Exit(1)

@app.command()
def check_outdated(
    days: int = typer.Option(
        7,
        "--days",
        "-d",
        help="Number of days after which a folder is considered outdated"
    ),
    config_path: Optional[Path] = CONFIG_PATH_OPTION,
    log_level: str = LOG_LEVEL_OPTION
):
    """Check which folders need organizing based on when they were last organized."""
    # Configure logger
    configure_logger(log_level)

    try:
        # Load configuration
        config = OrganizerConfig.load_config(config_path)

        # Create organizer and check outdated folders
        organizer = FileOrganizer(config)
        outdated_folders = organizer.check_outdated_folders(days)

        if not outdated_folders:
            console.print(f"[green]All folders are up-to-date (organized within the last {days} days)[/green]")
            return

        console.print(f"[yellow]Found {len(outdated_folders)} folders that haven't been organized in {days} days:[/yellow]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Folder", style="dim")
        table.add_column("Last Organized", style="cyan")
        table.add_column("Status", style="yellow")

        for folder in outdated_folders:
            if folder.last_organized:
                last_organized = folder.last_organized.strftime("%Y-%m-%d %H:%M:%S")
                days_ago = (datetime.now() - folder.last_organized).days
                status = f"{days_ago} days ago"
            else:
                last_organized = "Never"
                status = "Never organized"

            table.add_row(str(folder.path), last_organized, status)

        console.print(table)
        console.print("\n[bold]To organize these folders, run:[/bold]")
        console.print("  organizer organize --outdated-only")

    except Exception as e:
        logger.error(f"Failed to check outdated folders: {e}")
        raise typer.Exit(1)

@app.command()
def show_config(
    config_path: Optional[Path] = CONFIG_PATH_OPTION,
    log_level: str = LOG_LEVEL_OPTION
):
    """Show current configuration."""
    # Configure logger
    configure_logger(log_level)

    try:
        # Load configuration
        config = OrganizerConfig.load_config(config_path)

        actual_config_path = config_path or Path.home() / ".organizer.rc"
        console.print(f"[bold]Configuration file:[/bold] {actual_config_path}")
        console.print(f"[bold]Target folder:[/bold] {config.target_folder}")
        console.print(f"[bold]AI model:[/bold] {config.ai_config.model_type}")

        if config.ai_config.model_name:
            console.print(f"[bold]Model name:[/bold] {config.ai_config.model_name}")

        # Show source folders
        console.print("\n[bold]Source folders:[/bold]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Folder", style="dim")
        table.add_column("Patterns", style="cyan")
        table.add_column("Max Depth", style="cyan")
        table.add_column("Last Organized", style="green")

        for folder in config.source_folders:
            if folder.last_organized:
                last_organized = folder.last_organized.strftime("%Y-%m-%d %H:%M:%S")
            else:
                last_organized = "Never"

            table.add_row(
                str(folder.path),
                ", ".join(folder.patterns),
                str(folder.max_depth),
                last_organized
            )

        console.print(table)

    except Exception as e:
        logger.error(f"Failed to show configuration: {e}")
        raise typer.Exit(1)

@app.command()
def about(
    log_level: str = LOG_LEVEL_OPTION
):
    """Display information about Organizer and its author."""
    # Configure logger
    configure_logger(log_level)

    # Current year for copyright
    current_year = datetime.now().year

    # ASCII art banner
    banner = """
    ╔═══════════════════════════════════════════════╗
    ║                                               ║
    ║   ██████╗ ██████╗  ██████╗  █████╗ ███╗   ██╗ ║
    ║  ██╔═══██╗██╔══██╗██╔════╝ ██╔══██╗████╗  ██║ ║
    ║  ██║   ██║██████╔╝██║  ███╗███████║██╔██╗ ██║ ║
    ║  ██║   ██║██╔══██╗██║   ██║██╔══██║██║╚██╗██║ ║
    ║  ╚██████╔╝██║  ██║╚██████╔╝██║  ██║██║ ╚████║ ║
    ║   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝ ║
    ║                         THE AI FILE ORGANIZER  ║
    ║                                               ║
    ╚═══════════════════════════════════════════════╝
    """

    # Print banner with random color
    import random
    colors = ["bright_blue", "bright_green", "bright_magenta", "bright_cyan", "bright_yellow"]
    color = random.choice(colors)
    console.print(banner, style=color)

    # Print author info
    console.print("\n[bold]Built with ❤️  by:[/bold]")
    console.print("  [bold cyan]Alan Synn[/bold cyan] ([cyan]alan@alansynn.com[/cyan])")
    console.print(f"\n© {current_year} Alan Synn. All rights reserved.")
    console.print("\n[italic]\"A clean filesystem is a happy filesystem.\"[/italic]")

    # Print version info
    try:
        import importlib.metadata
        version = importlib.metadata.version("organizer")
        console.print(f"\n[dim]Version: {version}[/dim]")
    except:
        # If we can't get version info, just skip it
        pass

    # Add a little encouragement
    encouragements = [
        "Thank you for using Organizer!",
        "Your files will thank you for the organization!",
        "May your folders always be tidy!",
        "Keep calm and organize on!",
        "Turning chaos into order, one file at a time."
    ]
    console.print(f"\n[green]{random.choice(encouragements)}[/green]")

def check_environment():
    """Check and setup environment before running."""
    # Check for virtual environments
    in_venv = False
    venv_type = None

    # Check for standard venv
    if sys.prefix != sys.base_prefix:
        in_venv = True
        venv_type = "venv"

    # Check for Conda
    if os.environ.get('CONDA_PREFIX'):
        in_venv = True
        venv_type = "conda"

    if not in_venv:
        logger.warning("Not running in a virtual environment. This may lead to dependency issues.")
    else:
        logger.debug(f"Running in {venv_type} environment")

    return True

def main():
    """Main entry point with environment checks."""
    try:
        check_environment()
        app()
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()