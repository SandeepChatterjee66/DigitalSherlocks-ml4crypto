

# Trained Models
Here's the `trained_models.md` file that includes the requested links and instructions on how to load the models and perform inference on a new dataset.
This project includes two pre-trained models:

1. [GPT Best Model](https://drive.google.com/file/d/1ONYv5-Ga25Rn3JVWeWNQ28XccGKaU-_n/view?usp=drive_link)
2. [Final Best Model](https://drive.google.com/file/d/14uWjBUuH_ZeIjqO7kecX7eUUnisuS4li/view?usp=sharing)

## Loading the Models and Performing Inference

The models were trained using a fine-tuned BERT model for sequence classification. To use these models, follow the steps below.

### 1. Load the Pre-trained Model

To load the models for inference, use the following Python code:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from peft import get_peft_model, LoraConfig

# Set device (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the pre-trained model (choose between GPT or Final Best Model)
model_path = 'path_to_downloaded_model'  # Replace with the path to the downloaded model file
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
model.to(device)

# LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=8,  # Rank of low-rank matrices
    lora_alpha=32,  # Scaling factor for LoRA layers
    target_modules=["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"], 
    lora_dropout=0.1,  # Dropout for LoRA layers
    bias="none",  # No bias term in LoRA layers
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.to(device)

# Tokenizer example for new input text
def preprocess_and_predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move the input to the same device as the model
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Model inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get predictions (binary classification)
    predictions = torch.argmax(logits, dim=-1)

    # Return the predicted label
    return predictions.item()

# Example usage:
text = "Your input text here"
predicted_label = preprocess_and_predict(text)
print(f"Predicted Label: {predicted_label}")
```

### 2. Prepare Your Dataset for Inference

Your dataset should be in a CSV file with a column containing the binary strings. Here's how you can load and preprocess a new dataset:

```python
import pandas as pd

# Load your new dataset
df = pd.read_csv('path_to_your_new_dataset.csv')  # Replace with your dataset file path

# Assume the binary strings are in the 'Bitstream' column
binary_strings = df['Bitstream'].tolist()

# Preprocess the binary strings (assuming the XOR preprocessing function is defined)
def xor_preprocess(binary_string):
    s1 = binary_string[:512]
    s2 = binary_string[512:]
    s1_int = int(s1, 2)
    s2_int = int(s2, 2)
    xor_result = s1_int ^ s2_int
    xor_binary_string = format(xor_result, '0512b')  # Adjusted to ensure 512 bits
    return xor_binary_string

# Apply preprocessing to the binary strings
processed_binary_strings = [xor_preprocess(s) for s in binary_strings]

# Perform inference on each binary string
for binary_string in processed_binary_strings:
    predicted_label = preprocess_and_predict(binary_string)
    print(f"Predicted Label: {predicted_label}")
```

### 3. Downloading the Models

To download the models, follow these steps:
- [Download GPT Best Model](https://drive.google.com/file/d/1ONYv5-Ga25Rn3JVWeWNQ28XccGKaU-_n/view?usp=drive_link)
- [Download Final Best Model](https://drive.google.com/file/d/14uWjBUuH_ZeIjqO7kecX7eUUnisuS4li/view?usp=sharing)

Make sure to place the model files in the correct path and replace `'path_to_downloaded_model'` in the code with the correct path where you store the model.

---

Feel free to modify the code according to your dataset and model paths.


This Markdown file included:
- Links to download the GPT and Final Best Models.
- Code snippets to load the pre-trained models, process new input data, and perform inference using the fine-tuned BERT model.
Go to the main page of this repository for getting the other files and resources, or the overview of the project
