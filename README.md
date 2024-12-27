# Enhancing Software Defect Prediction Using a Sliding Window Approach with Pretrained Language Models (PLMs)


To fine tune either UnixCoder or CodeBERT and apply sliding window in inference stage, just
```bash
python finetune_BERT_based.py --output_dir=./bert1 --model_type roberta --model_name_or_path=<model huggingface name> --tokenizer_name=<model huggingface name>

```
To fine tune CodeT5 and apply sliding window in inference stage, just
```bash
python finetune_CodeT5.py --visible_gpu <GPU> --output_dir=./base1
```
