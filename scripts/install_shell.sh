#!/bin/bash
# Shell Integration Script for Organizer
# This script installs organizer globally and sets up shell integration

set -e  # Exit on error

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
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

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed. Please install pip first.${NC}"
    exit 1
fi

# Check if we're in the organizer directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${YELLOW}pyproject.toml not found in the current directory.${NC}"
    echo "Please navigate to the organizer project directory and try again."
    exit 1
fi

# Install organizer globally
echo -e "${BLUE}Installing organizer globally with pip...${NC}"
pip install -e .

# Verify installation
if ! command -v organizer &> /dev/null; then
    echo -e "${RED}Installation failed: organizer command not found.${NC}"
    echo "Check if your Python scripts directory is in your PATH."
    exit 1
fi

echo -e "${GREEN}Organizer successfully installed!${NC}"

# Create shell function and aliases
SHELL_FUNCTION=$(cat << 'EOF'

# Organizer function and aliases
alias org="organizer"
alias orgnow="organizer --now"
alias orgplan="organizer --dry-run"
alias orgauto="organizer --yes"
EOF
)

# Set up auto-completion
if [ "$SHELL_TYPE" = "bash" ]; then
    COMPLETION_SCRIPT=$(cat << 'EOF'

# Organizer auto-completion
_ORGANIZER_COMPLETE=bash_source organizer --install-completion &>/dev/null
EOF
    )
elif [ "$SHELL_TYPE" = "zsh" ]; then
    COMPLETION_SCRIPT=$(cat << 'EOF'

# Organizer auto-completion
_ORGANIZER_COMPLETE=zsh_source organizer --install-completion &>/dev/null
EOF
    )
fi

# Add to RC file
echo "Adding Organizer aliases to shell configuration file ($RC_FILE)."

# Check if already installed
if grep -q "# Organizer function and aliases" "$RC_FILE"; then
    echo -e "${YELLOW}Organizer aliases already exist in $RC_FILE.${NC}"
    echo "Would you like to overwrite the existing configuration? (y/n)"
    read -r OVERWRITE
    if [[ "$OVERWRITE" =~ ^[Yy]$ ]]; then
        # Remove existing configuration
        sed -i.bak '/# Organizer function and aliases/,/alias orgauto=/d' "$RC_FILE"
        sed -i.bak '/# Organizer auto-completion/,/_ORGANIZER_COMPLETE=/d' "$RC_FILE"
        echo -e "${GREEN}Existing configuration removed.${NC}"
    else
        echo -e "${YELLOW}Installation cancelled.${NC}"
        exit 0
    fi
fi

# Add new configuration
echo "" >> "$RC_FILE"
echo "$SHELL_FUNCTION" >> "$RC_FILE"
echo "$COMPLETION_SCRIPT" >> "$RC_FILE"

echo -e "${GREEN}Shell integration complete!${NC}"
echo -e "${YELLOW}To apply the settings, run:${NC}"
echo -e "${BLUE}source $RC_FILE${NC}"
echo ""
echo -e "${GREEN}Usage:${NC}"
echo "organizer         # Run organizer command"
echo "org               # Alias for organizer"
echo "orgnow            # Organize current directory (--now option)"
echo "orgplan           # Show organization plan only (--dry-run option)"
echo "orgauto           # Auto-execute without confirmation (--yes option)"
echo ""
echo -e "${GREEN}Auto-completion:${NC}"
echo "Organizer commands now support tab completion for commands and options."
echo "Try typing 'organizer ' and press TAB to see available commands."