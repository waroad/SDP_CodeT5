# Enhancing Software Defect Prediction Using a Sliding Window Approach with Pretrained Language Models (PLMs)


To fine tune CodeBERT and apply sliding window in inference stage, just
```bash
python finetune_CodeBERT_slide.py --output_dir=./bert1 --model_type roberta --model_name_or_path=microsoft/codebert-base --tokenizer_name=microsoft/codebert-base 

```
To fine tune CodeT5 and apply sliding window in inference stage, just
```bash
python finetune_CodeT5.py --visible_gpu <GPU> --output_dir=./base1
```