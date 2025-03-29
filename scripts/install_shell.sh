#!/bin/bash
# Shell Integration Script for Organizer
# This script sets up the organizer command to be used in your shell environment.

set -e  # Exit on error

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check supported shells
if [ -n "$BASH_VERSION" ]; then
    SHELL_TYPE="bash"
    RC_FILE="$HOME/.bashrc"
elif [ -n "$ZSH_VERSION" ]; then
    SHELL_TYPE="zsh"
    RC_FILE="$HOME/.zshrc"
else
    SHELL_TYPE="unknown"
    echo "Cannot detect your current shell. Please select the shell to install for:"
    select SHELL_CHOICE in "bash" "zsh"; do
        case $SHELL_CHOICE in
            bash)
                SHELL_TYPE="bash"
                RC_FILE="$HOME/.bashrc"
                break
                ;;
            zsh)
                SHELL_TYPE="zsh"
                RC_FILE="$HOME/.zshrc"
                break
                ;;
        esac
    done
fi

echo -e "${GREEN}Installing organizer for $SHELL_TYPE shell.${NC}"

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${YELLOW}Poetry is not installed. Using standard installation.${NC}"
    USE_POETRY=0
else
    echo -e "${GREEN}Poetry is installed.${NC}"
    echo "Would you like to use Poetry to run organizer? (y/n)"
    read -r USE_POETRY_ANSWER
    if [[ "$USE_POETRY_ANSWER" =~ ^[Yy]$ ]]; then
        USE_POETRY=1
    else
        USE_POETRY=0
    fi
fi

# Check current directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${YELLOW}pyproject.toml not found in the current directory.${NC}"
    echo "Please navigate to the organizer project directory and try again."
    exit 1
fi

# Choose installation method
if [ $USE_POETRY -eq 1 ]; then
    echo -e "${BLUE}Installing with Poetry...${NC}"
    poetry install

    # Find Poetry virtual environment path
    VENV_PATH=$(poetry env info --path)
    echo -e "${GREEN}Poetry virtual environment path: $VENV_PATH${NC}"
else
    echo -e "${BLUE}Installing with pip...${NC}"
    pip install -e .
fi

# Create shell function and aliases
SHELL_FUNCTION=$(cat << 'EOF'

# Organizer function
organize() {
    # Check if using Poetry
    if [ "$ORGANIZER_USE_POETRY" = "1" ]; then
        command poetry run organizer "$@"
    else
        # Use globally installed organizer
        command organizer "$@"
    fi
}

# Convenient aliases
alias org="organize"
alias orgnow="organize --now"
alias orgplan="organize --dry-run"
alias orgauto="organize --yes"
EOF
)

# Set up auto-completion
if [ "$SHELL_TYPE" = "bash" ]; then
    COMPLETION_SCRIPT=$(cat << 'EOF'

# Organizer auto-completion
_ORGANIZER_COMPLETE=bash_source organize --install-completion &>/dev/null
EOF
    )
elif [ "$SHELL_TYPE" = "zsh" ]; then
    COMPLETION_SCRIPT=$(cat << 'EOF'

# Organizer auto-completion
_ORGANIZER_COMPLETE=zsh_source organize --install-completion &>/dev/null
EOF
    )
fi

# Set up environment variables
if [ $USE_POETRY -eq 1 ]; then
    PROJECT_DIR=$(pwd)
    ENV_VARS=$(cat << EOF

# Organizer settings (auto-generated)
export ORGANIZER_USE_POETRY=1
export ORGANIZER_POETRY_PROJECT="$PROJECT_DIR"
EOF
    )
else
    # Find virtual environment path
    if [ -d ".venv" ]; then
        VENV_PATH="$(pwd)/.venv"
    elif [ -d "venv" ]; then
        VENV_PATH="$(pwd)/venv"
    else
        VENV_PATH=""
    fi

    if [ -n "$VENV_PATH" ]; then
        ENV_VARS=$(cat << EOF

# Organizer settings (auto-generated)
export ORGANIZER_USE_POETRY=0
export ORGANIZER_VENV_PATH="$VENV_PATH"
EOF
        )
    else
        ENV_VARS=$(cat << EOF

# Organizer settings (auto-generated)
export ORGANIZER_USE_POETRY=0
EOF
        )
    fi
fi

# Add to RC file
echo "Adding Organizer function to shell configuration file ($RC_FILE)."

# Check if already installed
if grep -q "# Organizer function" "$RC_FILE"; then
    echo -e "${YELLOW}Organizer function already exists in $RC_FILE.${NC}"
    echo "Would you like to overwrite the existing configuration? (y/n)"
    read -r OVERWRITE
    if [[ "$OVERWRITE" =~ ^[Yy]$ ]]; then
        # Remove existing configuration
        sed -i.bak '/# Organizer function/,/alias orgauto=/d' "$RC_FILE"
        sed -i.bak '/# Organizer settings (auto-generated)/,/ORGANIZER_VENV_PATH=/d' "$RC_FILE"
        sed -i.bak '/# Organizer settings (auto-generated)/,/ORGANIZER_POETRY_PROJECT=/d' "$RC_FILE"
        sed -i.bak '/# Organizer auto-completion/,/_ORGANIZER_COMPLETE=/d' "$RC_FILE"
        echo -e "${GREEN}Existing configuration removed.${NC}"
    else
        echo -e "${YELLOW}Installation cancelled.${NC}"
        exit 0
    fi
fi

# Add new configuration
echo "" >> "$RC_FILE"
echo "$ENV_VARS" >> "$RC_FILE"
echo "$SHELL_FUNCTION" >> "$RC_FILE"
echo "$COMPLETION_SCRIPT" >> "$RC_FILE"

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}To apply the settings, run:${NC}"
echo -e "${BLUE}source $RC_FILE${NC}"
echo ""
echo -e "${GREEN}Usage:${NC}"
echo "organize           # Standard file organization"
echo "org                # Alias for organize"
echo "orgnow             # Organize current directory (--now option)"
echo "orgplan            # Show organization plan only (--dry-run option)"
echo "orgauto            # Auto-execute without confirmation (--yes option)"
echo ""
echo -e "${GREEN}Auto-completion:${NC}"
echo "Organizer commands now support tab completion for commands and options."
echo "Try typing 'organize ' and press TAB to see available commands."