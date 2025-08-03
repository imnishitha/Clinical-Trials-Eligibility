import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import toml
import shutil
from datetime import datetime
from huggingface_hub import HfApi

def train(model, config, train_loader, val_loader, device, ):
    # Log Losses via Weights and Biases
    wandb.init(project='NLP_Project_Clinical_Trials', config=config)

    model.to(device)
    # Cross Entropy Loss is best for Multiclass classification
    loss_fn = nn.CrossEntropyLoss()

    # Adam Optimizer is a good starting point
    optimizer = optim.Adam(params=model.parameters(), lr=config["training"]["learning_rate"])

    num_epochs = config["training"]["num_epochs"]
    patience = config["training"].get("patience", 3) # Stops after 3 bad epochs if not in config
    alpha = config["training"].get("alpha", 1e-6) # Stops after 3 bad epochs if not in config
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader: 
            data = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()       # Reset the gradients from the previous batch
            output = model(data, attention_mask)
            loss = loss_fn(output, label)
            loss.backward()             # Compute gradients using backpropagation
            optimizer.step()

            train_loss += loss.item()
            print(f"Epoch: {epoch+1}, Train Loss: {loss.item()}")
        avg_train_loss = train_loss/len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():           # Don't need gradients while computing validation loss
            for batch in val_loader: 
                data = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label = batch["label"].to(device)
                output = model(data, attention_mask)
                val_loss += loss_fn(output, label).item()
                pred = output.argmax(dim=1)       # Get the predicted class
                correct += (pred == label).sum().item()
            
        avg_val_loss = val_loss/len(val_loader.dataset)
        accuracy = (correct/len(val_loader.dataset))*100
        print(f"\nEpoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f} %)\n")

        wandb.log(data={
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": accuracy
        })

        # Early Stopping
        
        if (best_val_loss - avg_val_loss) > alpha:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No Significant Improvement: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early Stopping Triggered!\nTraining Stopped after {epoch+1} epochs")
            break

    # Save the model on HuggingFace Hub - rdhopate
    save_dir = "Trained_Models"
    curr_time = str(datetime.now().strftime("%d%m%y_%H%M%S"))
    os.makedirs(name=save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"encoder_classifier_{curr_time}.bin")
    config_path = os.path.join(save_dir, "config.toml")
    
    torch.save(model.state_dict(), model_path)
    with open(config_path, "w") as f:
        toml.dump(config, f)

    api = HfApi(token=os.getenv("HF_TOKEN"))
    repo_id = "rdhopate/nlp-clinical-trials"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=model_path,
                    path_in_repo=f"encoder_classifier_{curr_time}.bin", 
                    repo_id=repo_id, 
                    repo_type="model")
    api.upload_file(path_or_fileobj=config_path,
                    path_in_repo="config.toml", 
                    repo_id=repo_id, 
                    repo_type="model")
    api.upload_file(path_or_fileobj="./BPE/bpe_tokenizer.json",
                    path_in_repo="tokenizer.json",
                    repo_id=repo_id,
                    repo_type="model")
    
    print(f"Model and Config File Uploaded to HuggingFace Hub at repo_id: {repo_id}")
    shutil.rmtree(path=save_dir)
    print(f"Folder {save_dir} Deleted Locally.")

def evaluate(model, test_data, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_data: 
            data = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            output = model(data, attention_mask)
            loss_sum += loss_fn(output, label).item()

            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
        
        avg_test_loss = loss_sum/len(test_data)
        accuracy = (correct/total)*100
        return avg_test_loss, accuracy

    
