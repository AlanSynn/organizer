#!/bin/bash
# Sample setup script for Organizer

# Check if we're in a virtual environment
check_venv() {
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "Running in standard virtual environment"
        return 0
    elif [ -n "$CONDA_PREFIX" ]; then
        echo "Running in conda environment"
        return 0
    else
        echo "Not running in a virtual environment"
        return 1
    fi
}

# Check if poetry is installed and use it, otherwise try pip
if command -v poetry &> /dev/null; then
    echo "Poetry found, using it for installation"
    INSTALL_CMD="poetry install"
    RUN_PREFIX="poetry run"
elif command -v pip &> /dev/null; then
    echo "Pip found, using it for installation"

    if ! check_venv; then
        echo "Warning: Not in a virtual environment. It's recommended to create one first."
        echo "You can create one with:"
        echo "  python -m venv .venv"
        echo "  source .venv/bin/activate  # On Unix/macOS"
        echo "  .venv\\Scripts\\activate    # On Windows"
        echo ""
        read -p "Continue without virtual environment? (y/n): " answer
        if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
            echo "Exiting..."
            exit 1
        fi
    fi

    INSTALL_CMD="pip install -e ."
    RUN_PREFIX=""
else
    echo "Neither Poetry nor Pip found. Please install one of them first."
    exit 1
fi

# Install project dependencies
echo "Installing project dependencies..."
$INSTALL_CMD

# Set up environment
echo "Setting up environment..."
mkdir -p ~/Organized

# Get custom config path or use default
echo "Where would you like to store the config file? (default: ~/.organizer.rc)"
read CONFIG_PATH
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH=~/.organizer.rc
    CONFIG_ARG=""
else
    CONFIG_ARG="--config $CONFIG_PATH"
fi

# Get model type
echo "Which AI model would you like to use? (gemini/llama)"
read MODEL_TYPE
MODEL_TYPE=${MODEL_TYPE:-gemini}

if [ "$MODEL_TYPE" = "gemini" ]; then
    # Gemini API setup
    echo "Please enter your Gemini API key:"
    read API_KEY

    if [ -n "$API_KEY" ]; then
        echo "Enter specific model name (default: gemini-pro):"
        read MODEL_NAME
        MODEL_NAME_ARG=""
        if [ -n "$MODEL_NAME" ]; then
            MODEL_NAME_ARG="--model-name $MODEL_NAME"
        fi

        echo "Initializing organizer with Gemini API..."
        $RUN_PREFIX organizer init --target ~/Organized --model gemini --api-key "$API_KEY" $MODEL_NAME_ARG $CONFIG_ARG

        # Add example source folders
        echo "Adding example source folders..."
        $RUN_PREFIX organizer add-source ~/Downloads --pattern "*.pdf" "*.docx" "*.xlsx" $CONFIG_ARG
        $RUN_PREFIX organizer add-source ~/Documents --pattern "*.txt" "*.md" "*.py" "*.json" $CONFIG_ARG

        echo "Setup complete! Try organizing your files with:"
        echo "$RUN_PREFIX organizer organize --dry-run $CONFIG_ARG"
    else
        echo "No API key provided. You'll need to manually initialize organizer."
    fi
elif [ "$MODEL_TYPE" = "llama" ]; then
    # Llama model setup
    echo "Please enter the path to your Llama model file (.gguf format):"
    read MODEL_PATH

    if [ -n "$MODEL_PATH" ]; then
        echo "Initializing organizer with Llama model..."
        $RUN_PREFIX organizer init --target ~/Organized --model llama --model-path "$MODEL_PATH" $CONFIG_ARG

        # Add example source folders
        echo "Adding example source folders..."
        $RUN_PREFIX organizer add-source ~/Downloads --pattern "*.pdf" "*.docx" "*.xlsx" $CONFIG_ARG
        $RUN_PREFIX organizer add-source ~/Documents --pattern "*.txt" "*.md" "*.py" "*.json" $CONFIG_ARG

        echo "Setup complete! Try organizing your files with:"
        echo "$RUN_PREFIX organizer organize --dry-run $CONFIG_ARG"
    else
        echo "No model path provided. You'll need to manually initialize organizer."
    fi
else
    echo "Unsupported model type: $MODEL_TYPE"
    echo "Please choose 'gemini' or 'llama'"
fi

# Show current configuration
echo "Current configuration:"
$RUN_PREFIX organizer show-config $CONFIG_ARG

# Explain usage with planning
echo ""
echo "How to use organizer with planning:"
echo "  $RUN_PREFIX organizer organize $CONFIG_ARG           # Creates a plan and asks for confirmation"
echo "  $RUN_PREFIX organizer organize --plan-only $CONFIG_ARG  # Only shows the plan without executing"
echo "  $RUN_PREFIX organizer organize --yes $CONFIG_ARG     # Executes without asking for confirmation"