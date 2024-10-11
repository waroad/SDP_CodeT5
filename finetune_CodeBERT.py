import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
start = time.time()


with open('train.jsonl', encoding="utf-8") as f:
    data=[]
    for idx, line in enumerate(f):
        line = line.strip()
        js = json.loads(line)
        data.append(js)
train_df = pd.DataFrame(data)
with open('test.jsonl', encoding="utf-8") as f:
    data = []
    for idx, line in enumerate(f):
        line = line.strip()
        js = json.loads(line)
        data.append(js)
test_df = pd.DataFrame(data)
with open('val.jsonl', encoding="utf-8") as f:
    data = []
    for idx, line in enumerate(f):
        line = line.strip()
        js = json.loads(line)
        data.append(js)
val_df = pd.DataFrame(data)

train_df=train_df[:]
val_df=val_df[:]
test_df=test_df[:]

# Check the split sizes
print(f"Training dataset size: {len(train_df)}")
print(f"Validation dataset size: {len(val_df)}")
print(f"Test dataset size: {len(test_df)}")


# Define a custom dataset
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['func']  # Use 'Description' as input
        target = self.data.iloc[index]['target']  # Use 'Severity' as output (integer 2-5)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': torch.tensor(target, dtype=torch.long)  # Integer target
        }


for ind,dev in enumerate(["devign1","devign2","devign3","devign4","devign5"]):
    # Parameters
    BATCH_SIZE = 16
    MAX_LEN = 512
    EPOCHS = 15
    LEARNING_RATE = 1e-5
    PATIENCE = 10  # Early stopping patience


    class BertClassifier(nn.Module):
        def __init__(self, bert_model_name, num_classes):
            super(BertClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(bert_model_name)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1]
            logits = self.classifier(pooled_output)
            return logits


    SEED = 1234+ind  # You can set any number you like
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # If you're using a GPU

    # Ensure deterministic behavior on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Initialize the tokenizer & model
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = BertClassifier('microsoft/codebert-base',2)

    # Create DataLoader
    train_dataset = TextDataset(train_df, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for d in train_loader:
        print(d['input_ids'].shape, d['attention_mask'].shape, d['targets'])
        break
    val_dataset = TextDataset(val_df, tokenizer, MAX_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    test_dataset = TextDataset(test_df, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()


    # Training function
    def train_epoch(model, data_loader, optimizer, device, scheduler):
        model = model.train()
        losses = []
        progress_bar = tqdm(data_loader, desc="Training", leave=False)

        for d in progress_bar:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)  # Shift targets from [2,5] to [0,3] for classification

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

        return losses


    # Evaluation function
    def eval_model(model, data_loader, device):
        model = model.eval()
        predictions = []
        real_values = []
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for d in progress_bar:
                input_ids = d['input_ids'].to(device)
                attention_mask = d['attention_mask'].to(device)
                targets = d['targets'].to(device)   # Shift targets for evaluation

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()   # Shift back to 2-5 range
                predictions.append(preds)
                real_values.append(targets.cpu().numpy())

        predictions = np.concatenate(predictions)
        real_values = np.concatenate(real_values)

        return predictions, real_values


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop with early stopping
    best_val_acc = 0
    best_val_f1 = 0
    epochs_no_improve = 0

    model_save_path = dev
    os.makedirs(model_save_path, exist_ok=True)
    tokenizer.save_pretrained(model_save_path)

    # Training loop
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        train_losses = train_epoch(model, train_loader, optimizer, device, scheduler)
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f'Training loss: {avg_train_loss}')

        val_preds, val_targets = eval_model(model, val_loader, device)
        # Calculate metrics
        val_f1 = f1_score(val_targets, val_preds, average='binary')
        val_precision = precision_score(val_targets, val_preds, average='binary')
        val_recall = recall_score(val_targets, val_preds, average='binary')
        val_accuracy = accuracy_score(val_targets, val_preds)

        for i in range(min(20, len(val_preds))):  # Print predictions and actual values for the first 10 samples
            print(f"pred:   {val_preds[i]} actual: {val_targets[i]}")
        print(f'Validation F1 Score: {val_f1}')
        print(f'Validation Precision: {val_precision}')
        print(f'Validation Recall: {val_recall}')
        print(f'Validation Accuracy: {val_accuracy}')

        # Early stopping check
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{model_save_path}/acc_best_model_state_dict.pth")
        else:
            epochs_no_improve += 1
            print("no")

        if epochs_no_improve == PATIENCE:
            print('Early stopping triggered')
            break


    # Inference start
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = BertClassifier('microsoft/codebert-base',2)
    model.load_state_dict(torch.load(f"{model_save_path}/acc_best_model_state_dict.pth", map_location=torch.device('cpu')))
    model.to(device)  # Move the model to the appropriate device
    model.eval()


    # Perform inference on the test set
    test_preds, test_targets = eval_model(model, test_loader, device)
    test_preds = (test_preds > 0.5).astype(int)
    test_f1 = f1_score(test_targets, test_preds, average='binary')
    test_precision = precision_score(test_targets, test_preds, average='binary')
    test_recall = recall_score(test_targets, test_preds, average='binary')
    test_accuracy = accuracy_score(test_targets, test_preds)

    print(f'Test F1 Score: {test_f1}')
    print(f'Test Precision: {test_precision}')
    print(f'Test Recall: {test_recall}')
    print(f'Test Accuracy: {test_accuracy}')

    # Display the results for 50 test samples
    # for i in range(min(50, len(test_preds))):
    #     print(f"pred:   {test_preds[i]}\nactual: {test_targets[i]}")

end = time.time()

print(f"{end - start:.5f} sec")