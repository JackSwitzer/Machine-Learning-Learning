#Training a Kolmogorov-Arnold Network for MNIST, final test evaluation accuracy of 97.73%
# First Dev project with cursor, 2024-09-05
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from kan import MultKAN
import os
from torch.optim.lr_scheduler import LambdaLR
import sys
import time
import threading
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# GPU availability check, TESTING
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available(): 
#     print("CUDA device count:", torch.cuda.device_count())
#     print("CUDA device name:", torch.cuda.get_device_name(0))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")


# Hyperparameters
input_size = 784  # 28x28 pixels
hidden_size = 128
output_size = 10  # 10 classes for MNIST
batch_size = 1024  # or 1024, depending on your GPU memory
learning_rate = 0.001
num_epochs = 9
warmup_epochs = 3
decay_epochs = warmup_epochs
decay_factor = 0.1

# KAN specific parameters
num_layers = 3
hidden_dim = 128
num_heads = 4  # New parameter for MultKAN

# convert all data loading into a function
def load_data():
    # Load and preprocess the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4), DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

# Model information loading and storage
def save_model(model, optimizer, epoch, loss, accuracy, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, filename)

def load_model(model, optimizer, filename):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        print(f"Loaded model from epoch {epoch} with accuracy {accuracy:.2f}%")
        return epoch, loss, accuracy
    return 0, None, None

# Learning rate scheduler with warmup and decay
def get_lr_scheduler(optimizer, warmup_epochs, decay_epochs, decay_factor):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return decay_factor ** ((epoch - warmup_epochs) // decay_epochs)
    
    return LambdaLR(optimizer, lr_lambda)

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.view(data.shape[0], -1).to(device)
            targets = targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):
    model.to(device)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    final_loss = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        data_time = 0
        backward_time = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data_start = time.time()
            data = data.view(data.shape[0], -1).to(device)
            targets = targets.to(device)
            data_time += time.time() - data_start

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(data)
                loss = criterion(outputs, targets)

            # Backward pass and optimization with mixed precision
            optimizer.zero_grad()
            backward_start = time.time()  # Add this line
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            backward_time += time.time() - backward_start

        final_loss = loss.item()
        
        scheduler.step()

        # Evaluate the model
        accuracy = evaluate_model(model, test_loader, device)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {final_loss:.4f}, Accuracy: {accuracy:.2f}%, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_time:.2f} seconds')
        print(f'Data time: {data_time:.2f}s, Backward time: {backward_time:.2f}s')
        
        save_model(model, optimizer, epoch + 1, final_loss, accuracy, 'kan_mnist_checkpoint.pth')

    return model, final_loss

def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor

def spinner(stop_event):
    spinner = spinning_cursor()
    while not stop_event.is_set():
        sys.stdout.write(next(spinner))
        sys.stdout.flush()
        sys.stdout.write('\b')
        time.sleep(0.1)

# Load the trained model
def load_trained_model(filename):
    checkpoint = torch.load(filename, map_location=device)
    model = MultKAN(
        width=[input_size, hidden_dim, hidden_dim, output_size],
        grid=5,
        k=3,
        mult_arity=num_heads,
        base_fun='silu',
        device=device
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch'], checkpoint['accuracy']

# Load and preprocess the MNIST test dataset
def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model on a subset of the test data
def evaluate_model_performance(model, test_loader, num_samples=1000):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            if total >= num_samples:
                break
            data = data.view(data.shape[0], -1).to(device)
            targets = targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, all_preds, all_targets

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Train Code
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=(stop_spinner,))
    spinner_thread.start()

    try:
        print("Starting main execution...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Using device: {device}")
        
        print("Loading data...")
        train_loader, test_loader = load_data()
        print("Data loaded successfully.")

        print("Initializing MultKAN model...")
        # Initialize the MultKAN model
        model = MultKAN(
            width=[input_size, hidden_dim, hidden_dim, output_size],
            grid=5,  # You may want to adjust this value
            k=3,  # You may want to adjust this value
            mult_arity=num_heads,
            base_fun='silu',
            device=device
        ).to(device)
        print("MultKAN model initialized.")

        criterion = nn.CrossEntropyLoss()  # Move criterion to GPU
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = get_lr_scheduler(optimizer, warmup_epochs, decay_epochs, decay_factor)

        print("Loading model checkpoint...")
        # Load the model if a checkpoint exists
        start_epoch, loaded_loss, loaded_accuracy = load_model(model, optimizer, 'kan_mnist_checkpoint.pth')
        print(f"Starting from epoch {start_epoch}")

        print("Starting model training...")
        # Train the model
        model, final_loss = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs - start_epoch, device)

        print("Performing final evaluation...")
        # Final evaluation
        final_accuracy = evaluate_model(model, test_loader, device)
        print(f'Final accuracy on the test set: {final_accuracy:.2f}%')

        print("Saving final model...")
        # Save the final model with accuracy and loss
        save_model(model, optimizer, num_epochs, final_loss, final_accuracy, 'kan_mnist_final.pth')

        print("Execution completed.")

    finally:
        stop_spinner.set()
        spinner_thread.join()


    # Visualization Code
    # Load the trained model
    model, epochs, final_accuracy = load_trained_model('kan_mnist_final.pth')
    print(f"Loaded model trained for {epochs} epochs with final accuracy: {final_accuracy:.2f}%")

    # Load test data
    test_loader = load_test_data()

    # Evaluate the model
    accuracy, all_preds, all_targets = evaluate_model_performance(model, test_loader)
    print(f"Model accuracy on test set: {accuracy:.2f}%")

    # Plot confusion matrix
    classes = list(range(10))  # 0-9 for MNIST
    plot_confusion_matrix(all_targets, all_preds, classes)
    print("Confusion matrix saved as 'confusion_matrix.png'")

    # Plot some example predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        data, target = next(iter(test_loader))
        data = data[i].view(1, -1).to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        ax.imshow(data.cpu().view(28, 28), cmap='gray')
        ax.set_title(f"True: {target[i]}, Pred: {pred.item()}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('example_predictions.png')
    plt.close()
    print("Example predictions saved as 'example_predictions.png'")
