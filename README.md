# Kimi_VL_Finetuning_Scripts

```
chmod +x *.sh
```

```
python finetune_kimi_vl.py \
  --images_dir "data/iam_dataset/images" \
  --annotations "data/iam_dataset/annotations.csv" \
  --model_name "moonshotai/Kimi-VL-A3B-Instruct" \
  --run_training
```

```
./export_kimi_vl.sh \
   --model_path "moonshotai/Kimi-VL-A3B-Instruct" \
   --adapter_path "saves/kimi_vl/lora/sft" \
   --export_dir "merged_model"
```

```
python inference_kimi_vl.py \
  --model_path "merged_model" \
  --image_folder "data/iam_dataset/images" \
  --output_csv "ocr_results.csv"
```