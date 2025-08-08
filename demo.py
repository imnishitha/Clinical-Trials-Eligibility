"""
Welcome to the shorter version of our project!
This demo file uses our models and provides a nice comparison on the unseen usecases for patient summaries and eligibility criteria.
"""

import torch
import torch.nn.functional as F
import toml
import string
import json
import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from PyTorch_Files.encoder_model import Classifier

# INSTANTIATE CONSTANTS
REPO_ID = "rdhopate/nlp-clinical-trials"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG FETCHING
config_path = hf_hub_download(repo_id=REPO_ID, filename="config.toml")
config = toml.load(config_path)
print("Config file fetching complete")

# TOKENIZER FETCHING
tokenizer_path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()
print(f"Tokenizer loaded with vocab size - {vocab_size}")

# MODEL FETCHING
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

# rnn_model = 
print("Model Initialised")

# STATE DICT - ENCODER
encoder_weights_path = hf_hub_download(repo_id=REPO_ID, filename="encoder_classifier_070825_144722.bin")
encoder_state_dict = torch.load(encoder_weights_path, map_location="cpu")
encoder_model.load_state_dict(state_dict=encoder_state_dict, strict=True)
encoder_model.to(DEVICE).eval()
print("Encoder Model State Dictionary loaded and ready for evaluation")

# STATE DICT - RNN
# rnn_weights_path = hf_hub_download(repo_id=REPO_ID, filename="rnn_encoder_classifier_030825_230954.bin")
# rnn_state_dict = torch.load(encoder_weights_path, map_location="cpu")
# rnn_model.load_state_dict(state_dict=encoder_state_dict, strict=True)
# rnn_model.to(DEVICE).eval()
# print("RNN Model State Dictionary loaded and ready for evaluation")

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
    for idx, subdata in enumerate(data):
        print(f"Example {idx+1}")
        patient = subdata['patient']
        criteria = subdata['criteria']
        sample_text = "[CLS] " + clean_text(criteria) + " [SEP] " + clean_text(patient)
        label, probabilities = predict_label(sample_text, model=encoder_model)
        print("Encoder Predicted label:", label)
        print("Encoder Probabilities:", probabilities)
        print(f"True label: {subdata['label']}")
        # label, probabilities = predict_label(sample_text, model=rnn_model)
        # print("Encoder Predicted label:", label)
        # print("RNN Probabilities:", probabilities)
        print('-'*100)