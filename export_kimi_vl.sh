#!/bin/bash
# export_kimi_vl.sh - Script for exporting fine-tuned KIMI-VL model

set -e  # Exit on any error

# Parse command line arguments
MODEL_PATH=""
ADAPTER_PATH=""
EXPORT_DIR=""

print_usage() {
    echo "Usage: $0 --model_path <model_path> --adapter_path <adapter_path> --export_dir <export_dir>"
    echo ""
    echo "Arguments:"
    echo "  --model_path    Path to the base model (e.g., moonshotai/Kimi-VL-A3B-Instruct)"
    echo "  --adapter_path  Path to the fine-tuned LoRA adapter"
    echo "  --export_dir    Directory to save the merged model"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --adapter_path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --export_dir)
            EXPORT_DIR="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$MODEL_PATH" ] || [ -z "$ADAPTER_PATH" ] || [ -z "$EXPORT_DIR" ]; then
    echo "Error: Missing required arguments"
    print_usage
    exit 1
fi

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: Virtual environment not activated."
    echo "It's recommended to run this in a virtual environment."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create export directory if it doesn't exist
mkdir -p "$EXPORT_DIR"

echo "Starting model export process..."
echo "Base model: $MODEL_PATH"
echo "Adapter path: $ADAPTER_PATH"
echo "Export directory: $EXPORT_DIR"

# Try different ways to merge the model
echo "Attempting to merge model..."

# Method 1: Try llamafactory-cli
if command -v llamafactory-cli &> /dev/null; then
    echo "Method 1: Using llamafactory-cli"
    llamafactory-cli export \
        --model_name_or_path "$MODEL_PATH" \
        --adapter_name_or_path "$ADAPTER_PATH" \
        --export_dir "$EXPORT_DIR" \
        --trust_remote_code
else
    echo "llamafactory-cli not found, trying alternative methods..."
    
    # Method 2: Try using Python module directly
    echo "Method 2: Using Python module"
    python -m llmtuner.export \
        --model_name_or_path "$MODEL_PATH" \
        --adapter_name_or_path "$ADAPTER_PATH" \
        --export_dir "$EXPORT_DIR" \
        --trust_remote_code
fi

# Check if export was successful
if [ $? -eq 0 ] && [ -d "$EXPORT_DIR" ] && [ "$(ls -A "$EXPORT_DIR")" ]; then
    echo "✅ Model export successful!"
    echo "Merged model saved to: $EXPORT_DIR"
    echo ""
    echo "To run inference with the exported model:"
    echo "python inference_kimi_vl.py --model_path \"$EXPORT_DIR\" --image_folder \"path/to/images\""
else
    echo "❌ Model export failed. Check the error messages above."
    exit 1
fi