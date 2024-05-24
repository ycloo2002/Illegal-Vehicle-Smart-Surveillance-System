import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.datasets import ImageFolder

# Define transformations for training and testing
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    # Load your dataset
    train_dataset = ImageFolder(root="F:/fyp_system/dataset/colour/train/", transform=transform_train)
    val_dataset = ImageFolder(root="F:/fyp_system/dataset/colour/val/", transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the pre-defined GoogLeNet model
    weights = None  # Set this to a specific weight if needed, e.g., weights='IMAGENET1K_V1'
    model = models.googlenet(weights=weights, aux_logits=True, init_weights=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # Adjust the final layer to match the number of classes
    model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, len(train_dataset.classes))
    model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, len(train_dataset.classes))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    def train(model, train_loader, criterion, optimizer, epoch):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, aux1, aux2 = model(inputs)
            loss1 = criterion(outputs, targets)
            loss2 = criterion(aux1, targets)
            loss3 = criterion(aux2, targets)
            loss = loss1 + 0.3 * (loss2 + loss3)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    def evaluate(model, val_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')
        
    def evaluate(model, val_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')
        
    num_epochs = 20
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, epoch)
        evaluate(model, val_loader)


    # Save the model
    model_path = "colour.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")