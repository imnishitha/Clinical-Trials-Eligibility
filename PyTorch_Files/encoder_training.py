import torch
import torch.nn as nn
import torch.optim as optim
import wandb

def train(model, config, train_loader, val_loader, device):
    # Log Losses via Weights and Biases
    wandb.init(project='NLP_Project_Clinical_Trials', config=config)

    model.to(device)
    # Cross Entropy Loss is best for Multiclass classification
    loss_fn = nn.CrossEntropyLoss()

    # Adam Optimizer is a good starting point
    optimizer = optim.Adam(params=model.parameters(), lr=config["training"]["learning_rate"])

    num_epochs = config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, label in train_loader: 
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()       # Reset the gradients from the previous batch
            output = model(data)
            loss = loss_fn(output, label)
            loss.backward()             # Compute gradients using backpropagation
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss/len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():           # Don't need gradients while computing validation loss
            for data, label in val_loader:
                output = model(data)
                val_loss += loss_fn(output, label).item()
                pred = output.argmax(dim=1, keepdim=True)       # Get the predicted class
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


def evaluate(model, test_data, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, label in test_data:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss_sum += loss_fn(output, label).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred == label).sum().item()
            total += label.size(0)
        
        avg_test_loss = loss_sum/len(test_data)
        accuracy = (correct/total)*100
        return avg_test_loss, accuracy

    
