# 🗂️ Organizer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/poetry-package-blueviolet.svg)](https://python-poetry.org/)

> 🤖 AI-powered file organization CLI tool that automatically categorizes and arranges files into meaningful hierarchies.
>
> 💻 Built with ❤️ by [Alan Synn](mailto:alan@alansynn.com)

---

## ✨ Features

- 🧠 **Intelligent Categorization** - Uses AI to analyze files and create meaningful folder structures
- 📂 **Organized Hierarchy** - Creates up to 3 levels of folder organization
- 🔄 **Duplicate Handling** - Smartly identifies and manages duplicate files
- 📊 **Incremental Organization** - Only processes new files to save time and resources
- 📱 **Multiple Models** - Supports Google Gemini (cloud) and Llama (local) models
- 📄 **Document Analysis** - Extracts content from PDFs and Word documents for better categorization
- 🐚 **Shell Integration** - Convenient command aliases for bash and zsh

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Key Features](#-key-features)
- [Shell Integration](#-shell-integration)
- [Document Content Analysis](#-document-content-analysis)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Poetry (recommended) or Pip
- Virtual environment (venv or conda)

### Install with Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/AlanSynn/organizer.git
cd organizer

# Install with Poetry
poetry install

# Activate virtual environment
poetry shell
```

### Install with pip

```bash
# Option 1: Install directly from GitHub
pip install git+https://github.com/AlanSynn/organizer.git

# Option 2: Clone and install locally
git clone https://github.com/AlanSynn/organizer.git
cd organizer
pip install -e .
```

### Quick Setup Script

```bash
# Make the script executable
chmod +x scripts/setup_example.sh

# Run the setup script
./scripts/setup_example.sh
```

The setup script will:
- Detect your environment (poetry, pip, venv, conda)
- Install dependencies
- Guide you through initial configuration
- Set up example source folders

---

## 🚀 Quick Start

```bash
# Initialize with Google Gemini (cloud model)
organizer init --target ~/Organized --model gemini --api-key YOUR_API_KEY

# Add a source folder to monitor
organizer add-source ~/Downloads

# Organize files (creates a plan and asks for confirmation)
organizer organize

# Organize files in the current directory
organizer organize --now
```

---

## ⚙️ Configuration

Organizer stores configuration in `~/.organizer.rc` by default.

### Google Gemini Setup

```bash
# Basic setup
organizer init --target ~/Organized --model gemini --api-key YOUR_API_KEY

# Advanced setup with specific model
organizer init --target ~/Organized --model gemini --model-name gemini-pro --api-key YOUR_API_KEY
```

### Local Llama Setup

```bash
# Basic setup
organizer init --target ~/Organized --model llama --model-path /path/to/your/model.gguf

# Custom model parameters
organizer init --target ~/Organized --model llama --model-path /path/to/model.gguf \
  --model-params '{"n_ctx":4096,"n_threads":8}'
```

### Add Source Folders

```bash
# Add a source folder with default settings
organizer add-source ~/Downloads

# Add a source folder with specific file patterns
organizer add-source ~/Documents --pattern "*.pdf" "*.docx" --max-depth 2
```

---

## 🧩 Usage Examples

### Basic Organization

```bash
# Generate a plan and ask for confirmation
organizer organize

# Create a plan without executing it
organizer organize --plan-only

# Skip confirmation
organizer organize --yes

# Test run (no changes)
organizer organize --dry-run
```

### Advanced Options

```bash
# Only organize outdated folders (not organized in 7 days)
organizer organize --outdated-only

# Change the outdated threshold
organizer organize --outdated-only --days 14

# Organize files within the current directory
organizer organize --now

# Include files that have already been organized
organizer organize --include-organized

# Combine options
organizer organize --now --yes --incremental
```

### Status and Reporting

```bash
# Check which folders are outdated
organizer check-outdated

# Show current configuration and status
organizer show-config

# Display information about the tool and its author
organizer about
```

### Logging Control

```bash
# Control logging verbosity
organizer organize --log-level debug    # Highly detailed logging
organizer organize --log-level info     # More detailed than default
organizer organize --log-level warning  # Default: warnings and errors
organizer organize --log-level error    # Only error messages
```

---

## 🔑 Key Features

### 📊 Intelligent Categorization

Organizer uses AI to analyze filenames and file contents to determine the most appropriate categorization:
- Creates up to 3 levels of folder hierarchy
- Suggests descriptive filenames while preserving original extensions
- Handles files based on both name and content analysis

### 🔄 Smart Duplicate Handling

- Detects true duplicates by comparing filename and file size
- Moves duplicates to a "duplicated items" folder in the source directory
- Preserves original files for review
- Prevents redundant processing of duplicate files

### 📱 In-Place Organization

The `--now` option organizes files within the current directory:
- Uses current directory as both source and target
- Creates appropriate folder hierarchies in-place
- Perfect for quick organization without configuration
- Can be combined with other options like `--incremental` and `--yes`

### ⏱️ Skip Already Organized Files

By default, Organizer skips previously organized files:
- Detects files already processed based on existing structure
- Saves time and reduces API calls by focusing on new files
- Can be disabled with `--include-organized` flag
- Works with all organization commands including `--now` and `--outdated-only`

### 📈 Outdated Folder Tracking

- Tracks when each folder was last organized
- Identifies folders not organized within a set period (default: 7 days)
- Allows selective organization of only outdated folders
- Provides visual reports showing outdated folders with timestamps

---

## 🐚 Shell Integration

Integrate Organizer with your shell environment (bash, zsh) for convenient access from anywhere.

### Automatic Installation

```bash
# In the project directory
./scripts/install_shell.sh
```

This script:
- Detects your current shell (bash or zsh)
- Checks for Poetry installation and configures accordingly
- Adds necessary functions to your shell configuration
- Sets up useful aliases
- Enables command auto-completion

### Manual Installation

Add to your `.bashrc` or `.zshrc`:

```bash
# Organizer function
organize() {
  if command -v organizer &> /dev/null; then
    organizer "$@"
  elif [ -d "venv" ] || [ -d ".venv" ]; then
    # Run in virtual environment
    if [ -d "venv" ]; then
      source venv/bin/activate
    else
      source .venv/bin/activate
    fi
    organizer "$@"
    deactivate
  fi
}

# Convenient aliases
alias org="organize"
alias orgnow="organize --now"
alias orgplan="organize --dry-run"

# Enable auto-completion (Bash)
_ORGANIZER_COMPLETE=bash_source organize --install-completion > /dev/null 2>&1

# For Zsh, use this instead:
# _ORGANIZER_COMPLETE=zsh_source organize --install-completion > /dev/null 2>&1
```

### Usage After Shell Integration

```bash
organize           # Standard file organization
org                # Alias for organize
orgnow             # Organize current directory
orgplan            # Show organization plan only
orgauto            # Run without confirmation
```

### Auto-Completion

After installation, Organizer commands support tab completion:

```bash
organize <TAB>     # Show available commands
organize init -<TAB>   # Show available options for init command
```

---

## 📄 Document Content Analysis

For document files, Organizer extracts content for more accurate categorization based on actual document contents rather than just filenames.

### Supported Document Types

- **PDF files (.pdf)** - Extracts text from the first few pages
- **Word documents (.docx, .doc)** - Extracts text from the document

This allows for much better organization of documents by topic and content, especially when filenames are not descriptive.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

MIT License

Copyright (c) 2023-2024 Alan Synn (alan@alansynn.com)