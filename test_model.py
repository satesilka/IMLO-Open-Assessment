import time
import torch
from torch import device
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# Define transformations for the test data
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the test dataset
test_dataset = datasets.Flowers102(root='data', split='test', download=True, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# Define the CNN model (ensure it matches the architecture used for training)
class FlowerCNN(nn.Module):
    def __init__(self):
        super(FlowerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.bn6 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 102)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 512 * 7 * 7)
        x = self.dropout(torch.relu(self.bn6(self.fc1(x))))
        x = self.fc2(x)
        return x


# Load the saved model weights
model = FlowerCNN()
model.load_state_dict(torch.load('flowers102_classifier_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define the criterion
criterion = nn.CrossEntropyLoss()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to evaluate accuracy
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    elapsed_time = time.time() - start_time

    print(f'Accuracy: {accuracy:.2f}%, Time: {elapsed_time:.2f}s')
    return accuracy, elapsed_time


# Assume test_loader is already defined and loaded with test data
test_accuracy = evaluate_model(model, test_loader)

