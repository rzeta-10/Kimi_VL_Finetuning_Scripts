#!/usr/bin/env python3
"""
Inference script for fine-tuned KIMI-VL model on OCR tasks

Usage:
python inference_kimi_vl.py --model_path /path/to/merged_model --image_folder /path/to/images
"""

import argparse
import os
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoProcessor

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned KIMI-VL model")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the merged model or model name on HuggingFace")
    parser.add_argument("--image_folder", type=str, required=True, 
                       help="Folder containing images for OCR")
    parser.add_argument("--output_csv", type=str, default="ocr_results.csv",
                       help="Output CSV file name")
    parser.add_argument("--prompt", type=str, default="Extract the text from this image.",
                       help="Prompt to use for extraction")
    parser.add_argument("--show_images", action="store_true",
                       help="Display images during inference")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of tokens to generate")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the model and processor
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if device == "cuda" else "auto",
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        print("Model and processor loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have enough VRAM")
        print("2. Try using quantization if memory is limited")
        print("3. Check that the model path exists and contains model files")
        return
    
    # Get device being used (for display)
    model_device = next(model.parameters()).device
    print(f"Model loaded on: {model_device}")
    
    # Get list of image files (supporting common formats)
    image_folder = Path(args.image_folder)
    if not image_folder.exists():
        print(f"Error: Image folder {image_folder} does not exist")
        return
        
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} images for processing")
    
    # List to store results
    results = []
    
    # Loop through each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        try:
            # Load the input image
            image = Image.open(image_path)
            
            # Visualize the image if requested
            if args.show_images:
                plt.figure(figsize=(8, 8))
                plt.imshow(image)
                plt.axis('off')
                plt.title(f"Input Image: {image_file}")
                plt.show()
            
            print(f"Processing: {image_file}")
            
            # Prepare the input for OCR using the Kimi-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": args.prompt}
                    ]
                }
            ]
            
            try:
                # Convert messages to model input format
                text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
                inputs = processor(images=[image], text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
                
                # Generate text
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                
                # Process the output
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                extracted_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                extracted_text = f"Error: {str(e)}"
            
            # Display the extracted text
            print("Extracted Text:")
            print(extracted_text)
            print("-" * 50)
            
            # Store the result
            results.append({
                "image_name": image_file,
                "ocr_output": extracted_text
            })
            
            # Clean up to save memory
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            results.append({
                "image_name": image_file,
                "ocr_output": f"Error: {str(e)}"
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()