import torch
import toml
from dataset_creation import train_loader, val_loader, test_loader, tokenizer
from encoder_model import Classifier
from encoder_training import train, evaluate

# Config File
config = toml.load('config.toml')

# Set Device - GPU Preferred for Parallel Optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get Model Parameters
vocab_size = tokenizer.get_vocab_size()

# Model Initialization
model = Classifier(
    vocab_size=vocab_size,
    max_len=config["model"]["max_len"],
    d_model=config["model"]["embeddim_dim"],
    d_k=config["model"]["d_k"],
    d_v=config["model"]["d_k"],
    n_heads=config["model"]["n_heads"],
    d_ff=config["model"]["hidden_layers"],
    n_layers=config["model"]["transformer_layers"],
    n_classes=config["model"]["num_classes"]
)

# Training the model
train(model=model, config=config, train_loader=train_loader, val_loader=val_loader, device=device)
print("Training Complete!")

# Test Set Evaluation
test_loss, accuracy = evaluate(model=model, test_data=test_loader, device=device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")