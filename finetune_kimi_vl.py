#!/usr/bin/env python3
"""
KIMI-VL Fine-tuning on IAM Handwriting Dataset

This script:
1. Processes custom OCR dataset
2. Generates training configuration
3. Fine-tunes KIMI-VL model using LLaMA-Factory
"""

import argparse
import json
import subprocess
import gc, torch; gc.collect()
from pathlib import Path
from typing import Dict
import wandb
from tqdm import tqdm

# ---------------------------------------------------------------------------#
# Helpers                                                                    #
# ---------------------------------------------------------------------------#

def _chatml_sample(img_name: str, prompt: str, answer: str) -> Dict:
    return {
        "images": [img_name],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + prompt.strip(),
            },
            {"from": "gpt", "value": answer.strip()},
        ],
    }


def build_dataset(annotations: Path, out_dir: Path, fmt: str,
                  p_col: str, a_col: str, images_dir: Path) -> int:
    """Convert CSV / JSONL to ChatML → `train.jsonl` with absolute image paths. Returns #samples."""
    out_path = out_dir / "train.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        if fmt == "csv":
            import csv
            with annotations.open(newline="", encoding="utf-8") as f:
                for row in tqdm(csv.DictReader(f), desc="CSV→JSONL"):
                    # Construct absolute path to the image
                    abs_img_path = str(images_dir.resolve() / row["filename"])
                    out_f.write(json.dumps(
                        _chatml_sample(abs_img_path, row[p_col], row[a_col]),
                        ensure_ascii=False) + "\n")
                    n += 1
        else:  # JSONL → JSONL
            with annotations.open(encoding="utf-8") as f:
                for line in tqdm(f, desc="JSONL→JSONL"):
                    rec = json.loads(line)
                    # Construct absolute path to the image
                    abs_img_path = str(images_dir.resolve() / rec["filename"])
                    out_f.write(json.dumps(
                        _chatml_sample(abs_img_path, rec[p_col], rec[a_col]),
                        ensure_ascii=False) + "\n")
                    n += 1
    return n


def patch_dataset_info(repo_root: Path, name: str, rel_path: str) -> None:
    """Ensure data/dataset_info.json contains an entry for this dataset."""
    info_path = repo_root / "data" / "dataset_info.json"
    info_path.parent.mkdir(exist_ok=True)
    info = json.loads(info_path.read_text("utf-8")) if info_path.exists() else {}

    if name not in info:
        info[name] = {
            "file_name":  rel_path,                 # "kimi_custom/train.jsonl"
            "formatting": "sharegpt",
            "columns": {                            # logical → physical
                "messages": "conversations",
                "images":   "images"
            }
        }
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False),
                             "utf-8")


def write_yaml(path: Path, model: str, dset: str, dset_dir: Path, ftype: str,
               out_dir: Path, bsz: int, accum: int, epochs: float,
               lr: float, res: int, run_name: str, lr_scheduler_type: str) -> None:
    yaml = f"""
### model
model_name_or_path: {model}
quantization_bit: 4
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: {ftype}

### dataset
dataset: {dset}
template: kimi_vl
cutoff_len: 2048

### output
output_dir: {out_dir}
logging_steps: 5
save_steps: 500
overwrite_output_dir: true
report_to: wandb

### train
per_device_train_batch_size: {bsz}
gradient_accumulation_steps: {accum}
learning_rate: {lr}
lr_scheduler_type: {lr_scheduler_type}
num_train_epochs: {epochs}
bf16: true
run_name: {run_name}
"""
    path.write_text(yaml.strip() + "\n")


# ---------------------------------------------------------------------------#
# Main                                                                       #
# ---------------------------------------------------------------------------#
def main() -> None:
    ap = argparse.ArgumentParser("Fine‑tune Kimi‑VL with LLaMA‑Factory")
    ap.add_argument("--images_dir", type=Path, required=True, 
                   help="Directory containing the images")
    ap.add_argument("--annotations", type=Path, required=True,
                   help="Path to CSV or JSONL file with annotations")
    ap.add_argument("--output_root", type=Path, default=Path("data/kimi_custom"),
                   help="Directory to store processed dataset")
    ap.add_argument("--annotation_format", choices=["csv", "jsonl"], default="csv",
                   help="Format of annotations file")
    ap.add_argument("--prompt_col", default="prompt",
                   help="Column name for prompts in annotations")
    ap.add_argument("--answer_col", default="answer",
                   help="Column name for answers in annotations")
    ap.add_argument("--model_name", default="moonshotai/Kimi-VL-A3B-Instruct",
                   help="Base model to fine-tune")
    ap.add_argument("--finetuning_type", choices=["lora", "qlora", "full"], default="lora",
                   help="Fine-tuning method")
    ap.add_argument("--batch_size", type=int, default=1,
                   help="Batch size per device")
    ap.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps")
    ap.add_argument("--epochs", type=float, default=30.0,
                   help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate")
    ap.add_argument("--resolution", type=int, default=224,
                   help="Image resolution")
    ap.add_argument("--run_training", action="store_true",
                   help="Whether to run training after setup")
    ap.add_argument("--lr_scheduler_type", type=str, default="linear",
                   help="Learning rate scheduler type (e.g., linear, cosine)")
    ap.add_argument("--run_name", type=str, default="kimi-vl-lora",
                   help="WandB run name")
    ap.add_argument("--use_wandb", action="store_true",
                   help="Enable WandB logging")
    args = ap.parse_args()

    if args.use_wandb and args.run_training:
        wandb.init(
            project="Kimi-VL-Finetune",
            name=args.run_name,
            config={
                "model": args.model_name,
                "finetuning_type": args.finetuning_type,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "lr_scheduler_type": args.lr_scheduler_type,
                "dataset": "kimi_custom"
            }
        )

    repo_root = Path(".").resolve()
    dset_dir = args.output_root
    dset_dir.mkdir(parents=True, exist_ok=True)

    # ---- images/ symlink or copy ------------------------------------------ #
    img_dst = dset_dir / "images"
    if not img_dst.exists():
        try:
            print(f"Creating symlink from {args.images_dir} to {img_dst}")
            img_dst.symlink_to(args.images_dir.resolve(), target_is_directory=True)
        except Exception as e:
            print(f"Failed to create symlink: {e}. Trying to copy directory...")
            import shutil
            shutil.copytree(args.images_dir, img_dst)
            print(f"Copied images from {args.images_dir} to {img_dst}")
    # ----------------------------------------------------------------------- #

    print("[1/4] Building dataset …")
    n = build_dataset(args.annotations, dset_dir,
                      args.annotation_format, args.prompt_col, args.answer_col, args.images_dir)
    print(f"   ✓ {n} samples → {dset_dir/'train.jsonl'}")

    print("[2/4] Patching dataset_info.json …")
    base_data = (repo_root / "data").resolve()
    
    # Create data directory if it doesn't exist
    base_data.mkdir(exist_ok=True)
    
    try:
        rel_path = str((dset_dir / "train.jsonl").resolve().relative_to(base_data))
    except ValueError:
        # If the path is not relative to base_data, use absolute path
        rel_path = str((dset_dir / "train.jsonl").resolve())
        print(f"Warning: Using absolute path {rel_path}")
    
    patch_dataset_info(repo_root, "kimi_custom", rel_path)
    print("   ✓ dataset_info.json updated")

    print("[3/4] Writing YAML …")
    yaml_path = repo_root / "train_kimi_custom.yaml"
    out_dir = repo_root / "saves/kimi_vl/lora/sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    write_yaml(yaml_path, args.model_name, "kimi_custom", dset_dir,
               args.finetuning_type, out_dir,
               args.batch_size, args.grad_accum, args.epochs, args.lr,
               args.resolution, args.run_name, args.lr_scheduler_type)
    print(f"   ✓ YAML saved → {yaml_path}")

    if args.run_training:
        print("[4/4] Launching training … (Ctrl‑C to abort)")
        # Try both llamafactory-cli and through Python module
        try:
            subprocess.run(["llamafactory-cli", "train", str(yaml_path)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error with llamafactory-cli: {e}")
            print("Trying alternative method...")
            
            # Fallback to direct Python module call
            try:
                cmd = ["python", "-m", "llmtuner.train.sft", "--config", str(yaml_path)]
                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Training failed: {e}")
                
        if args.use_wandb:
            wandb.finish()
    else:
        print("\nDone. To start training, run one of:")
        print(f"  llamafactory-cli train {yaml_path}")
        print(f"  python -m llmtuner.train.sft --config {yaml_path}")


if __name__ == "__main__":
    # Clean CUDA memory
    torch.cuda.empty_cache()
    gc.collect()
    main()