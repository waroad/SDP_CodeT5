# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import time

# import bleu
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, T5Config, T5ForConditionalGeneration)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    filename="train.log",
                    filemode='w',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    task_prefix = "Defect: "

    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = js['func']
            nl = 'true' if js['target'] == 1 else 'false'
            examples.append(
                Example(
                    idx=idx,
                    source=task_prefix + code,
                    target=nl,
                )
            )

    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    if stage == "test":
        args.max_source_length = 5120
    else:
        args.max_source_length = 512
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        # 加mask
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def sliding_window_batch_inference(model, tokenizer, input_ids, attention_mask, max_length=512, overlap=50):
    batch_predictions = [[] for _ in range(10)]
    for i in range(len(input_ids)):
        tokens = input_ids[i].tolist()
        token_len = len([t for t in tokens if t != tokenizer.pad_token_id])  # Exclude padding tokens
        # If token length is less than max_length, directly generate the prediction
        if token_len <= max_length:
            with torch.no_grad():
                pred = model.generate(input_ids=input_ids[i].unsqueeze(0),
                                      attention_mask=attention_mask[i].unsqueeze(0), max_length=128)
            text = tokenizer.decode(pred[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for b in batch_predictions:
                b.append(text)
            continue
        # Otherwise, split into sliding windows and perform inference on each
        window_predictions = []
        for start in range(0, token_len, max_length - overlap):
            if start + max_length > token_len:
                end = token_len
                start = token_len - max_length
            else:
                end = start + max_length
            window_input_ids = input_ids[i][start:end].unsqueeze(0).to(input_ids.device)
            window_attention_mask = torch.ones_like(window_input_ids).to(
                input_ids.device)  # Assuming no padding within the window
            with torch.no_grad():
                pred = model.generate(input_ids=window_input_ids, attention_mask=window_attention_mask, max_length=128)
            text = tokenizer.decode(pred[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            window_predictions.append(text)

        numeric_predictions = [1 if 'true' in pred else 0 for pred in window_predictions]
        for ind, b in enumerate(batch_predictions):
            final_prediction1 = sum(numeric_predictions) / len(numeric_predictions) >= ind * 0.05
            final_prediction1 = 'true' if final_prediction1 else 'false'
            b.append(final_prediction1)

    return batch_predictions


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=False,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    # Other parameters

    parser.add_argument("--train_filename", default="train.jsonl", type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default="val.jsonl", type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default="test.jsonl", type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=512, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=5, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true', default=False,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=False,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--visible_gpu', type=str, default="",
                        help="use how many gpus")

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    args.device = device

    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    model_config = T5Config.from_pretrained("Salesforce/codet5-small")
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small", config=model_config)
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    if args.local_rank != -1:
        # Distributed training
        try:
            # from apex.parallel import DistributedDataParallel as DDP
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)

    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    logger.info("model loaded!")

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)

        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')

        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=t_total)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss, best_acc = 0, 0, 0, 0, 0, 1e6, 0
        for epoch in range(args.num_train_epochs):

            bar = tqdm(train_dataloader, total=len(train_dataloader))

            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                labels = [
                    [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for
                    labels_example in target_ids
                ]
                labels = torch.tensor(labels).to(device)

                out = model(input_ids=source_ids, attention_mask=source_mask, labels=labels)
                loss = out.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                if (nb_tr_steps + 1) % 100 == 0:
                    logger.info("Epoch {}, step {}, train loss {}".format(epoch, nb_tr_steps, train_loss))
                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # Start Evaling model
                model.eval()
                eval_loss, tokens_num = 0, 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        labels = [
                            [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for
                            labels_example in target_ids
                        ]
                        labels = torch.tensor(labels).to(device)

                        tokens_num += torch.tensor(
                            [(labels_example != -100).sum().item() for labels_example in labels]).sum().item()

                        loss = model(input_ids=source_ids, attention_mask=source_mask, labels=labels).loss
                    eval_loss += loss.sum().item()
                # tokens_num += num.sum().item()
                # Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                # save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss < best_loss:
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    logger.info("  " + "*" * 20)
                    best_loss = eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                # Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                    # Calculate bleu
                    if 'dev_bleu' in dev_dataset:
                        eval_examples, eval_data = dev_dataset['dev_bleu']
                    else:
                        eval_examples = read_examples(args.dev_filename)
                        eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
                        eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                        all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                        eval_data = TensorDataset(all_source_ids, all_source_mask)
                        dev_dataset['dev_bleu'] = eval_examples, eval_data

                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    model.eval()
                    p = []
                    for batch in eval_dataloader:
                        batch = tuple(t.to(device) for t in batch)
                        source_ids, source_mask = batch
                        with torch.no_grad():
                            preds = model.generate(input_ids=source_ids, attention_mask=source_mask, max_length=128)

                            for pred in preds:
                                text = tokenizer.decode(pred, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
                                p.append(text)

                    model.train()
                    predictions = []
                    sum1 = 0
                    with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
                            os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
                        for ref, gold in zip(p, eval_examples):
                            predictions.append(str(gold.idx) + '\t' + ref)
                            f.write(str(gold.idx) + '\t' + ref + '\n')
                            f1.write(str(gold.idx) + '\t' + gold.target + '\n')
                            if ref == gold.target:
                                sum1 += 1
                    logger.info("Epoch {}, the accuracy is {}".format(epoch, sum1 / len(eval_examples)))

                    if sum1 / len(eval_examples) > best_acc:
                        best_acc = sum1 / len(eval_examples)
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_test:
        print("ho")
        model.load_state_dict(torch.load(f"./{args.output_dir}/checkpoint-best-acc/pytorch_model.bin"))
        files = []
        overlap1 = [50,100,150,200,250,300,350,400,450]
        # if args.dev_filename is not None:
        if args.test_filename is not None:
            for i in range(len(overlap1)):
                files.append(args.test_filename)
        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids, all_source_mask)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            ps = [[] for _ in range(10)]
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask = batch
                batch_predictions = sliding_window_batch_inference(model, tokenizer, source_ids, source_mask,
                                                                   overlap=overlap1[idx])
                # Collect the predictions
                for ind, p in enumerate(ps):
                    p.extend(batch_predictions[ind])

            model.train()
            predictions = []
            sum1 = 0
            for ind, p in enumerate(ps):
                with open(os.path.join(args.output_dir,
                                       "{}_{}_thresh_O{}.output".format(args.test_filename[:-6], str(overlap1[idx]),
                                                                       str(round(ind * 0.05, 2)))), 'w') as f, open(
                        os.path.join(args.output_dir,
                                     "{}_{}_thresh_O{}.gold".format(args.test_filename[:-6], str(overlap1[idx]),
                                                                   str(round(ind * 0.05, 2)))), 'w') as f1:
                    for ref, gold in zip(p, eval_examples):
                        predictions.append(str(gold.idx) + '\t' + ref)
                        f.write(str(gold.idx) + '\t' + ref + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target + '\n')
                        if ref == gold.target:
                            sum1 += 1
    # logger.info("Epoch {}, the accuracy is {}".format(epoch, sum1/len(eval_examples)))


if __name__ == "__main__":
    main()