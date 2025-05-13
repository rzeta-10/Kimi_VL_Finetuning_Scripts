#!/bin/bash
# prepare_iam_dataset.sh - Script for preparing IAM dataset

set -e  # Exit on any error

# Default paths
OUTPUT_DIR="data/iam_dataset"
ANNOTATIONS_FILE="$OUTPUT_DIR/annotations.csv"
IMAGES_DIR="$OUTPUT_DIR/images"

# Create directories
mkdir -p "$IMAGES_DIR"

# Function to display usage information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Prepare IAM dataset for KIMI-VL fine-tuning"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR     Set output directory (default: $OUTPUT_DIR)"
    echo "  --download           Attempt to download dataset from HuggingFace"
    echo "  --help               Display this help message"
}

# Parse command line options
DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --output-dir)
            OUTPUT_DIR="$2"
            ANNOTATIONS_FILE="$OUTPUT_DIR/annotations.csv"
            IMAGES_DIR="$OUTPUT_DIR/images"
            mkdir -p "$IMAGES_DIR"
            shift 2
            ;;
        --download)
            DOWNLOAD=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "Setting up IAM dataset in $OUTPUT_DIR"

# Attempt to download if requested
if $DOWNLOAD; then
    echo "Attempting to download IAM dataset from HuggingFace..."
    
    # Check if Python and required libraries are available
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 is required but not found"
        exit 1
    fi
    
    # Create and run a Python script to download the dataset
    DOWNLOAD_SCRIPT=$(cat << 'EOF'
from datasets import load_dataset
from pathlib import Path
import csv
from tqdm import tqdm
import sys

# Get output directory from command line
output_dir = Path(sys.argv[1])
images_dir = output_dir / "images"
annotations_file = output_dir / "annotations.csv"

try:
    print("Loading IAM dataset from Hugging Face...")
    dataset = load_dataset("gagan3012/IAM")
    
    # Create annotations CSV
    with open(annotations_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'prompt', 'answer'])
        
        # Process training split
        for idx, example in enumerate(tqdm(dataset['train'], desc="Processing samples")):
            # Get image and text
            image = example['image']
            text = example['text']
            
            # Save image to file
            image_filename = f"iam-{idx:05d}.jpg"
            image_path = images_dir / image_filename
            image.save(image_path)
            
            # Add to annotations
            writer.writerow([
                image_filename,
                "Extract the handwritten text from this image.",
                text
            ])
    
    print(f"Dataset prepared: {len(dataset['train'])} samples")
    print(f"- Images saved to: {images_dir}")
    print(f"- Annotations saved to: {annotations_file}")
    
except Exception as e:
    print(f"Error downloading dataset: {e}")
    sys.exit(1)
EOF
)

    # Execute the download script
    echo "$DOWNLOAD_SCRIPT" > /tmp/download_iam.py
    
    echo "Installing required Python packages..."
    pip install datasets tqdm
    
    echo "Downloading dataset..."
    python3 /tmp/download_iam.py "$OUTPUT_DIR"
    
    # Check if download was successful
    if [ $? -ne 0 ]; then
        echo "Download failed. Please prepare the dataset manually."
        create_sample_annotation
    fi
    
    rm /tmp/download_iam.py
else
    # Create a sample annotation file if not downloading
    if [ ! -f "$ANNOTATIONS_FILE" ]; then
        echo "Creating sample annotation file..."
        echo "filename,prompt,answer" > "$ANNOTATIONS_FILE"
        echo "sample.jpg,Extract the handwritten text from this image.,Sample text" >> "$ANNOTATIONS_FILE"
        
        echo "Created template annotation file at $ANNOTATIONS_FILE"
        echo ""
        echo "To use your own dataset:"
        echo "1. Place your images in $IMAGES_DIR"
        echo "2. Update $ANNOTATIONS_FILE with your image filenames and OCR ground truth"
        echo "3. Each row should have: filename,prompt,answer"
    else
        echo "Annotation file already exists at $ANNOTATIONS_FILE"
    fi
fi

echo ""
echo "Dataset preparation complete!"
echo ""
echo "Next steps:"
echo "1. Run fine-tuning script:"
echo "   python finetune_kimi_vl.py --images_dir \"$IMAGES_DIR\" --annotations \"$ANNOTATIONS_FILE\" --run_training"
echo ""
echo "For more options:"
echo "   python finetune_kimi_vl.py --help"