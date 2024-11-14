# -*- coding: utf-8 -*-
"""S.Chatterjee - DigitalSherlock

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Jj8CiWaaqDF9DfJv2UYGAAGWmj7T0jIA
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.


sandeepchatterjee66_ml4crypto_path = kagglehub.dataset_download('sandeepchatterjee66/ml4crypto')

print('Data source import complete.')



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("/content/TrainingData.csv")
data

len(data["Bitstream"][0])

# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
# import os

# # Check if TPU is available
# import os
# import transformers

# if 'COLAB_TPU_ADDR' in os.environ:
#     TPU = True
#     resolver = transformers.TPUMembershipFilter()
#     transformers.utils.set_seed(42)
# else:
#     TPU = False

# # Load and preprocess the data
# class BitstreamDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length=1024):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         bitstream = self.data['Bitstream'][idx]
#         label = self.data['class'][idx]
#         inputs = self.tokenizer.encode_plus(bitstream,
#                                            max_length=self.max_length,
#                                            pad_to_max_length=True,
#                                            return_tensors='pt')
#         return inputs, label

# # Fine-tune GPT on TPU
# if TPU:
#     import torch_xla
#     import torch_xla.core.xla_model as xm
#     import torch_xla.distributed.parallel_loader as pl

#     model = GPT2LMHeadModel.from_pretrained('gpt2')
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#     dataset = BitstreamDataset(data, tokenizer)
#     train_loader = pl.ParallelLoader(dataset, [xm.xla_device()])

#     optimizer = AdamW(model.parameters(), lr=2e-5)
#     scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                 num_warmup_steps=100,
#                                                 num_training_steps=len(train_loader) * 3)

#     model.train()
#     for epoch in range(3):
#         for inputs, labels in train_loader.per_device_loader(xm.xla_device()):
#             outputs = model(inputs, labels=labels)
#             loss = outputs.loss
#             xm.optimizer_step(optimizer)
#             scheduler.step()
#             xm.mark_step()

#     # Evaluate on test set
#     model.eval()
#     accuracy = 0
#     for inputs, labels in test_dataloader:
#         outputs = model(inputs)
#         predictions = outputs.logits.argmax(dim=1)
#         accuracy += (predictions == labels).float().mean()
#     print(f'Test accuracy: {accuracy / len(test_dataloader)}')
# else:
#     # Use CPU/GPU if TPU is not available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # Rest of the code remains the same as before

pip install torch

pip install peft

import torch

# Reload the best model
model.load_state_dict(torch.load(best_model_path))
model.to(device)

# Evaluation loop with reversed logits
model.eval()

# Assuming input_ids, attention_mask, and labels are already tensors:
# Ensure that all tensors are moved to the appropriate device (GPU or CPU)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
labels = labels.to(device)

# Reverse the logits function
def reverse_logits(logits):
    return logits * -1  # Reverse logits by multiplying by -1

# Test the model without reversing logits
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # Calculate accuracy without reversing logits
    correct_predictions = (predictions == labels).sum().item()
    accuracy_without_reverse = correct_predictions / len(labels) * 100

print(f"Accuracy without reversing logits on the entire dataset: {accuracy_without_reverse:.2f}%")

# Test the model with reversed logits
with torch.no_grad():
    reversed_logits = reverse_logits(logits)
    reversed_predictions = torch.argmax(reversed_logits, dim=-1)

    # Calculate accuracy with reversed logits
    correct_predictions = (reversed_predictions == labels).sum().item()
    accuracy_with_reverse = correct_predictions / len(labels) * 100

print(f"Accuracy with reversed logits on the entire dataset: {accuracy_with_reverse:.2f}%")

import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from peft import get_peft_model, LoraConfig, PeftModel

df = data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)

# Ensure the model's config uses the pad_token
model.config.pad_token_id = tokenizer.pad_token_id

# Resize embeddings for new pad token
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=8,  # rank of low-rank matrices (you can tune this)
    lora_alpha=32,  # scaling factor for LoRA layers
    target_modules=["attn.c_attn", "attn.c_proj"],  # which modules to apply LoRA to
    lora_dropout=0.1,  # dropout for LoRA layers
    bias="none",  # no bias term in LoRA layers
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.to(device)

# Load your dataset (adjust the file path and column names)
#df = pd.read_csv('/kaggle/input/ml4crypto/TrainingData.csv')  # Replace with your dataset file path

# Extract binary strings and labels
binary_strings = df['Bitstream'].tolist()
labels = df['class'].tolist()

# Preprocess the binary strings by splitting them into halves and XOR'ing the halves
def xor_preprocess(binary_string):
    # Split the string into two halves
    s1 = binary_string[:512]
    s2 = binary_string[512:]

    # XOR the halves
    s1_int = int(s1, 2)
    s2_int = int(s2, 2)
    xor_result = s1_int ^ s2_int

    # Convert XOR result back to binary string (512 bits)
    xor_binary_string = format(xor_result, '512b')
    return xor_binary_string

# Apply preprocessing to all binary strings
processed_binary_strings = [xor_preprocess(s) for s in binary_strings]

# Tokenize the processed binary strings
inputs = tokenizer(processed_binary_strings, padding=True, truncation=True, max_length=1024, return_tensors="pt")

# Convert to tensors and move to the appropriate device
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
labels = torch.tensor(labels).to(device)

# Create a DataLoader for batching
dataset = TensorDataset(input_ids, attention_mask, labels)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 15  # Update epoch count to 10 as per your request
best_accuracy = 0.0
best_model_path = "gpt_best_model.pth"

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        # Move the batch to the device
        input_ids, attention_mask, labels = [item.to(device) for item in batch]

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}")

    # Save the best model based on validation accuracy
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Calculate accuracy
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Save the model if it's the best
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with accuracy: {accuracy * 100:.2f}%")

# Reload the best model
model.load_state_dict(torch.load(best_model_path))
model.to(device)

# Evaluation loop with random sampling and reversed logits
model.eval()

# Test the best model on the entire validation dataset
model.eval()

# Initialize counters for accuracy
total_correct = 0
total_samples = 0

# Loop over the validation set
with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, labels = [item.to(device) for item in batch]

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Calculate accuracy
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

# Calculate final accuracy
accuracy = total_correct / total_samples * 100
print(f"Accuracy on the entire validation dataset: {accuracy:.2f}%")

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
from peft import get_peft_model, LoraConfig

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# LoRA configuration for efficient fine-tuning with BERT
lora_config = LoraConfig(
    r=8,  # rank of low-rank matrices
    lora_alpha=32,  # scaling factor for LoRA layers
    target_modules=["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"],  # BERT compatible modules
    lora_dropout=0.1,  # dropout for LoRA layers
    bias="none",  # no bias term in LoRA layers
)

# Apply LoRA to the model
try:
    model = get_peft_model(model, lora_config)
    model.to(device)
except ValueError as e:
    print(f"Error: {e}")
    print("Please check the target modules. Make sure they match the BERT model structure.")
    raise

# Load your dataset (adjust the file path and column names)
df = data  # Replace with your dataset file path

# Extract binary strings and labels
binary_strings = df['Bitstream'].tolist()
labels = df['class'].tolist()

# Preprocess the binary strings by splitting them into halves and XOR'ing the halves
def xor_preprocess(binary_string):
    s1 = binary_string[:512]
    s2 = binary_string[512:]
    s1_int = int(s1, 2)
    s2_int = int(s2, 2)
    xor_result = s1_int ^ s2_int
    xor_binary_string = format(xor_result, '0512b')  # Adjusted to ensure 512 bits
    return xor_binary_string

# Apply preprocessing to all binary strings
processed_binary_strings = [xor_preprocess(s) for s in binary_strings]

# Tokenize the processed binary strings
inputs = tokenizer(processed_binary_strings, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Convert to tensors and move to the appropriate device
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
labels = torch.tensor(labels).to(device)

# Create a DataLoader for batching
dataset = TensorDataset(input_ids, attention_mask, labels)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 20  # Update epoch count to 20
best_accuracy = 0.0
best_model_path = "best_model.pth"

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_mask, labels = [item.to(device) for item in batch]

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}")

    # Validation phase
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Calculate accuracy
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Save the best model based on validation accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with accuracy: {accuracy * 100:.2f}%")

# Reload the best model
model.load_state_dict(torch.load(best_model_path))
model.to(device)

# Final evaluation on the full dataset can now proceed here.

# Test the best model on the entire validation dataset
model.eval()

# Initialize counters for accuracy
total_correct = 0
total_samples = 0

# Loop over the validation set
with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, labels = [item.to(device) for item in batch]

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Calculate accuracy
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

# Calculate final accuracy
accuracy = total_correct / total_samples * 100
print(f"Accuracy on the entire validation dataset: {accuracy:.2f}%")

# Optionally: Reverse logits and calculate accuracy again
# def reverse_logits(logits):
#     return logits * -1  # Reverse logits by multiplying by -1

# # Test the model with reversed logits on the entire validation dataset
# total_correct_reversed = 0
# with torch.no_grad():
#     for batch in val_dataloader:
#         input_ids, attention_mask, labels = [item.to(device) for item in batch]

#         # Forward pass
#         outputs = model(input_ids, attention_mask=attention_mask)
#         logits = outputs.logits

#         # Reverse the logits
#         reversed_logits = reverse_logits(logits)
#         reversed_predictions = torch.argmax(reversed_logits, dim=-1)

#         # Calculate accuracy with reversed logits
#         total_correct_reversed += (reversed_predictions == labels).sum().item()

# # Calculate final accuracy with reversed logits
# accuracy_reversed = total_correct_reversed / total_samples * 100
# print(f"Accuracy with reversed logits on the entire validation dataset: {accuracy_reversed:.2f}%")