# DigitalSherlocks-ml4crypto
 In this work, we introduce an approach for classifying binary sequences by using transformer-based architectures with specialized preprocessing. Given a binary input sequence, we split it into halves and perform an XOR operation to differentiating bit patterns. We employ both BERT and GPT-2 models to predict classes from the resulting bitwise-processed sequences, adapting them through efficient parameter fine-tuning. This preprocessing and fine-tuning together reduces computational overhead while maintaining accuracy. Experimental results demonstrate the effectiveness of this approach on a binary classification task, with little advantage. Our findings open avenues for applying transformers in low-dimensional, binary-based classification scenarios, offering efficient, adaptable solutions for real-world applications. In future work, we can analyze if the advantage is increasing function in terms of training data and not advantage is not a neglible function in terms of security parameter (bit size).

 Please View this report for more details : [Report - Indistinguishability Adversary under Ciphertext-Only Attack](https://github.com/SandeepChatterjee66/DigitalSherlocks-ml4crypto/blob/main/Report%20-%20S_Chatterjee___Indistinguishability_Adversary_under_Ciphertext_Only_Attack.pdf)


# Digital Sherlock - Model Training and Inference

## Project Overview

The "Digital Sherlock" project focuses on applying state-of-the-art machine learning models, such as BERT and GPT, for sequence classification and inference tasks. The project involves training a BERT-based model and a GPT-based model, both fine-tuned for a specific task related to NLP. The trained models are evaluated on a dataset, and their performance is reported in a PDF file.

## Files Overview

### 1. `report.pdf`
This file contains the comprehensive report detailing the training process, results, and analysis of the models used in this project. It includes:

- Model architecture and configuration
- Training procedure and hyperparameters
- Performance metrics (accuracy, loss, etc.)
- Evaluation results and visualizations

This report serves as a detailed analysis of the model's performance and provides insights into areas of improvement.

### 2. `S_Chatterjee_DigitalSherlock.ipynb`
This Jupyter notebook contains the code used for model training, evaluation, and analysis. It includes:

- Data preprocessing steps to prepare the dataset for training.
- Model architecture setup for both BERT and GPT-based models.
- Training loops for fine-tuning the models on the dataset.
- Evaluation of the models on validation data, along with the results.
- Performance visualizations (loss and accuracy curves).

This notebook serves as the main script for reproducing the results in the report and can be used to train and evaluate the models on your own.

### 3. `trained_models.txt`
This text file contains a list of the trained models, their configurations, and checkpoints. Each model entry includes details such as:

- Model type (e.g., BERT, GPT)
- Hyperparameters used during training
- Directory or path to the saved model checkpoint

This file helps you keep track of different trained models and their specific configurations.

### 4. `TrainingData.csv`
This CSV file contains the dataset used for training and evaluating the models. It includes the following columns:

- `text`: The input text (e.g., sentences, paragraphs, or documents) that the models are trained to classify.
- `label`: The target label corresponding to each input text.

The dataset is used as the training and validation data for both the BERT and GPT models.

### 5. Trained Models
Downloading the Models

To download the models, follow these steps:
- [Download GPT Best Model](https://drive.google.com/file/d/1ONYv5-Ga25Rn3JVWeWNQ28XccGKaU-_n/view?usp=drive_link)
- [Download Final Best Model](https://drive.google.com/file/d/14uWjBUuH_ZeIjqO7kecX7eUUnisuS4li/view?usp=sharing)

### 5.1. `best_model` (BERT Model)
This file contains the best-performing BERT model checkpoint, which was fine-tuned for the sequence classification task. The model is saved in a format compatible with Hugging Face's Transformers library and can be loaded for inference or further fine-tuning.

- Model architecture: `BertForSequenceClassification`
- Fine-tuned for a specific downstream task (e.g., text classification)

### 5.2. `gpt_best_model`
This file contains the best-performing GPT-based model checkpoint, which was fine-tuned on the dataset for a sequence classification task. Like the BERT model, this model is saved in a format compatible with Hugging Face's GPT architecture and can be used for generating predictions or further fine-tuning.

- Model architecture: `GPT2LMHeadModel` (or another variant, depending on the task)
- Fine-tuned for a specific downstream task

## Getting Started

### Prerequisites

- Python 3.6+
- Required libraries (you can install them via `requirements.txt` or directly using pip):
  - `transformers`
  - `torch`
  - `pandas`
  - `matplotlib`
  - `sklearn`

You can install the required libraries with the following command:

```bash
pip install -r requirements.txt
```

### Running the Project

1. **Load and Preprocess Data**:
   Load the dataset from `dataset.csv` using pandas and preprocess it (tokenization, padding, etc.) before training the models. The preprocessing steps are implemented in the notebook `S_Chatterjee_DigitalSherlock.ipynb`.

2. **Train the Models**:
   In the notebook, run the code for training both BERT and GPT-based models. Ensure you have the correct environment set up and the GPU available for faster training.

3. **Evaluate the Models**:
   After training, evaluate the models on the validation set and save the best-performing models in `best_model` (BERT) and `gpt_best_model`.

4. **Review Results**:
   The final results and performance analysis can be found in `report.pdf`. This document provides a detailed evaluation of each model's performance.

### Loading Pre-trained Models for Inference

You can load the best-performing models (`best_model` for BERT and `gpt_best_model` for GPT) for inference or further tasks:

```python
from transformers import BertForSequenceClassification, GPT2LMHeadModel, BertTokenizer, GPT2Tokenizer

# Load BERT model
bert_model = BertForSequenceClassification.from_pretrained('path_to_best_model')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load GPT model
gpt_model = GPT2LMHeadModel.from_pretrained('path_to_gpt_best_model')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
```

### Example of Inference with the BERT Model

```python
# Example of BERT inference
text = "This is an example input sentence."
inputs = bert_tokenizer(text, return_tensors="pt")
outputs = bert_model(**inputs)
logits = outputs.logits
predictions = logits.argmax(dim=-1)
```

### Example of Inference with the GPT Model

```python
# Example of GPT inference
text = "Once upon a time"
inputs = gpt_tokenizer(text, return_tensors="pt")
outputs = gpt_model.generate(**inputs)
generated_text = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Conclusion

This repository provides all the necessary tools to replicate and analyze the sequence classification task using BERT and GPT models. It includes the trained models, dataset, evaluation results, and a detailed report. Feel free to experiment with the models and improve the performance by modifying hyperparameters or training configurations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### Explanation of the Sections:

1. **Project Overview**: Describes the project and its objectives, focusing on the use of BERT and GPT models for sequence classification.
2. **Files Overview**: Detailed descriptions of the four main files you mentioned (`report.pdf`, `S_Chatterjee_DigitalSherlock.ipynb`, `trained_models.txt`, `dataset.csv`, and the two models).
3. **Getting Started**: Instructions for setting up the project environment and running the project, including installing dependencies and running the training code.
4. **Running the Project**: Details on how to load data, train models, and evaluate them.
5. **Loading Pre-trained Models for Inference**: How to load and use the trained models for inference.
6. **Example Code**: Provides code snippets for running inference with both BERT and GPT models.
7. **License**: A section for specifying the project's license.

This `README.md` provides a clear, structured way to explain the project and guide users through setting up and running the code.
