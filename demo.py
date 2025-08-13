"""
Welcome to the shorter version of our project!
This demo file uses our models and provides a nice comparison on the unseen usecases for patient summaries and eligibility criteria.

Recommended Python Version:
    Python 3.10.17 — This script has been tested and works best with this version.
    Anticipating that the code works with Python>=3.8

Pre-requisites:
    - pip must be installed and accessible from your command line.
    - All dependencies listed in requirements.txt must be installed.
      You can install them by running:
          pip install -r requirements.txt

How to Run:
    1. Clone the repository:
        git clone https://github.com/imnishitha/Clinical-Trials-Eligibility
    2. Navigate to the project directory:
        cd Clinical-Trials-Eligibility
    3. Install dependencies:
        pip install -r requirements.txt
    4. Run the demo:
        python demo.py or python3 demo.py

Note:
    - Ensure that the repository structure remains intact, as the script relies on files 
      and modules in the PyTorch_Files folder.
    - The demo uses a small sample dataset (demo_data.json) included in the repository.
    - Encourage the user to modify/add examples in demo_data.json in the same format for additional testing.
"""

import torch
import torch.nn.functional as F
import toml
import string
import json
import numpy as np
from tabulate import tabulate
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from PyTorch_Files.encoder_model import Classifier
from PyTorch_Files.rnn_model import RNNClassifierFromScratch

# INSTANTIATE CONSTANTS
REPO_ID = "rdhopate/nlp-clinical-trials"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG FETCHING
print("Initializing HuggingFace Hub")
print("Initializing Config File")
config_path = hf_hub_download(repo_id=REPO_ID, filename="config.toml")
config = toml.load(config_path)
print("Config file fetching complete")

# TOKENIZER FETCHING
print("Initializing Tokenizer")
tokenizer_path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()
print(f"Tokenizer loaded with vocab size - {vocab_size}")

# MODEL FETCHING
print("Initializing Models")
encoder_model = Classifier(
    vocab_size=vocab_size,
    max_len=config["model"]["max_len"],
    d_model=config["model"]["embedding_dim"],
    d_k=config["model"]["d_k"],
    d_v=config["model"]["d_v"],
    n_heads=config["model"]["n_heads"],
    d_ff=config["model"]["hidden_layers"],
    n_layers=config["model"]["transformer_layers"],
    n_classes=config["model"]["num_classes"]
)

#rnn model
rnn_model = RNNClassifierFromScratch(
    vocab_size=vocab_size,
    embedding_dim=config["model"]["embedding_dim"],
    hidden_size=config["model"]["rnn"]["rnn_hidden_size"],
    num_layers=config["model"]["rnn"]["rnn_num_layers"],
    num_labels=config["model"]["num_classes"],
    dropout_rate=config["model"]["rnn"]["rnn_dropout"]
)
print("Models Initialized")

# STATE DICT - ENCODER
encoder_weights_path = hf_hub_download(repo_id=REPO_ID, filename="encoder_classifier_070825_144722.bin")
encoder_state_dict = torch.load(encoder_weights_path, map_location="cpu")
encoder_model.load_state_dict(state_dict=encoder_state_dict, strict=True)
encoder_model.to(DEVICE).eval()
print("Encoder Model State Dictionary loaded and ready for evaluation")

# STATE DICT - RNN
rnn_weights_path = hf_hub_download(repo_id=REPO_ID, filename="rnn_encoder_classifier_030825_230954.bin")
rnn_state_dict = torch.load(rnn_weights_path, map_location="cpu")
rnn_model.load_state_dict(state_dict=rnn_state_dict, strict=True)
rnn_model.to(DEVICE).eval()
print("RNN Model State Dictionary loaded and ready for evaluation")

# SAMPLE DATA
with open('demo_data.json') as file:
    data = json.load(file)

# INFERENCE
def label_map(label):
    maps = {0:"Negative", 1:"Neutral", 2:"Positive"}
    return maps[label]

def clean_text(text):
    text = text.lower()
    text = text.replace("-", " ")
    text = "".join([word for word in text if word not in string.punctuation])
    text = text.strip()
    return text

def encode_text(text):
    print(f"Input Text cleaned and consolidated with sample -> {text[:50]}")
    encoding = tokenizer.encode(text)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long)
    attention_mask = torch.tensor([[1]*len(encoding.ids)], dtype=torch.long)
    print("Text successfully converted to token ids and attention mask")
    return input_ids, attention_mask

def predict_label(text, model):
    input_ids, attention_mask = encode_text(text)
    input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).item()
    return label_map(pred), np.round(probs.squeeze().cpu().numpy(), 6)


if __name__ == "__main__":
    print(f"{'-'*20} Testing Examples {'-'*20}")
    
    table_data = []
    
    for idx, subdata in enumerate(data):
        patient = subdata['patient']
        criteria = subdata['criteria']
        sample_text = "[CLS] " + clean_text(criteria) + " [SEP] " + clean_text(patient)
        
        # Encoder Model Prediction
        label_encoder, probs_encoder = predict_label(sample_text, model=encoder_model)
        
        # RNN Model Prediction
        label_rnn, probs_rnn = predict_label(sample_text, model=rnn_model)
        
        table_data.append([
            f"Example {idx+1}",
            "Encoder",
            label_encoder,
            f"{subdata['label']}",
            f"{probs_encoder}"
        ])
        
        table_data.append([
            "", 
            "RNN",
            label_rnn,
            "", 
            f"{probs_rnn}"
        ])

    headers = ["Sample", "Model", "Predicted Label", "True Label", "Probabilities"]
    
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))