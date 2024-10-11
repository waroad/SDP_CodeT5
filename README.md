# Enhancing Software Defect Prediction Using a Sliding Window Approach with CodeT5


To fine tune a CodeBERT, just
```bash
python finetune_CodeBERT.py
```
To fine tune and apply sliding window in inference stage, just
```bash
python finetune_CodeT5.py --visible_gpu <GPU> --output_dir=./base1
```