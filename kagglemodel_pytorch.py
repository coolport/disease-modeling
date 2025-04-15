import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("Data source import complete.")


# Define functions for loading and resizing images
def load_and_resize_image(file_path, target_shape=(128, 128)):
    image = cv2.imread(file_path)
    resized_image = cv2.resize(image, target_shape)
    return resized_image


# Define the function to load each image class (target) stored by individual directory
def load_image_class_by_directory(image_dir):
    # Load and resize images
    image_files = os.listdir(image_dir)
    images = []
    for file in image_files:
        if file.endswith(".jpg") or file.endswith(".JPG"):
            image_path = os.path.join(image_dir, file)
            resized_image = load_and_resize_image(image_path)
            images.append(resized_image)

    print(f"Num of images: {len(images)}")
    print(f"Single image shape before flattening: {images[0].shape}")
    return images


# Display some images
def display_images(images, num_images_to_display=6):
    fig, axes = plt.subplots(1, num_images_to_display, figsize=(20, 5))
    for i in range(num_images_to_display):
        # Convert the image to a supported depth (e.g., CV_8U) before color conversion
        image = images[i].astype(np.uint8)
        axes[i].imshow(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )  # Convert BGR to RGB for displaying with matplotlib
        axes[i].axis("off")
    plt.show()


# Download dataset
nirmalsankalana_rice_leaf_disease_image_path = kagglehub.dataset_download(
    "nirmalsankalana/rice-leaf-disease-image"
)

# Load images from each class
base_path = "/home/aidan/.cache/kagglehub/datasets/nirmalsankalana/rice-leaf-disease-image/versions/1/"
images_Bacterialblight = load_image_class_by_directory(
    os.path.join(base_path, "Bacterialblight")
)
images_Blast = load_image_class_by_directory(os.path.join(base_path, "Blast"))
images_Brownspot = load_image_class_by_directory(os.path.join(base_path, "Brownspot"))
images_Tungro = load_image_class_by_directory(os.path.join(base_path, "Tungro"))

# Define class labels
classes = {"Bacterialblight": 0, "Blast": 1, "Brownspot": 2, "Tungro": 3}
inverted_classes = {0: "Bacterialblight", 1: "Blast", 2: "Brownspot", 3: "Tungro"}

images_lst_lst = [images_Bacterialblight, images_Blast, images_Brownspot, images_Tungro]
# Dictionary to store the number of image samples
classes_dict = {}
for i, images in enumerate(images_lst_lst):
    classes_dict.update({inverted_classes[i]: len(images)})
    print(f"Disease: {inverted_classes[i]} --- Images: {len(images)}")


# Create a PyTorch Dataset class
class RiceLeafDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label


# Split data into train and test sets
def split_train_test_files(images_lst_lst=[], num_test_set=int):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # Iterate through the image classes
    for class_idx, images in enumerate(images_lst_lst):
        test_set = images[:num_test_set]
        train_set = images[num_test_set:]

        # Create labels for each image
        test_labels_set = [class_idx] * len(test_set)
        train_labels_set = [class_idx] * len(train_set)

        train_images.extend(train_set)
        train_labels.extend(train_labels_set)
        test_images.extend(test_set)
        test_labels.extend(test_labels_set)

    return train_images, train_labels, test_images, test_labels


# Number of images to set aside as test set per class
num_test_set = 20

# Split the image files into train and test sets
train_images, train_labels, test_images, test_labels = split_train_test_files(
    images_lst_lst, num_test_set
)

# Define transformations for training data (including augmentation)
train_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(128, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Define transformations for validation and test data
val_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create datasets
train_dataset = RiceLeafDataset(train_images, train_labels, transform=train_transforms)

# Split the training data into training and validation sets
train_size = int(0.75 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Adjust validation dataset transforms
val_dataset.dataset.transform = val_transforms

# Create test dataset
test_dataset = RiceLeafDataset(test_images, test_labels, transform=val_transforms)

# Create data loaders
batch_size = 64
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Define CNN model
class RiceLeafCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(RiceLeafCNN, self).__init__()
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Function to compute accuracy
def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# Training function
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=2, device="cuda"
):
    model = model.to(device)
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                "/home/aidan/code/python/modeling/finals/rice_disease_detector_model.pth",
            )
            print(f"Model saved with validation accuracy: {val_acc:.2f}%")

    return model, history


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model, loss function, and optimizer
model = RiceLeafCNN(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Train the model
model, history = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=2, device=device
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Training Accuracy")
plt.plot(history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

# Evaluate model on test set
model.load_state_dict(
    torch.load(
        "/home/aidan/code/python/modeling/finals/rice_disease_detector_model.pth"
    )
)
test_accuracy = compute_accuracy(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")


# Function to make prediction on a single image
def predict_single_image(model, image, device):
    model.eval()

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = val_transforms
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    return prediction.item(), confidence.item()


# Function to display and predict an image
def make_predictions(image_class, image_idx):
    class_val = classes[image_class]

    # Get the single image
    image = test_images[class_val * num_test_set + image_idx]

    # Display image
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Make prediction
    pred_class, confidence = predict_single_image(model, image, device)

    print(f"Actual: {image_class}")
    print(f"Predicted: {inverted_classes[pred_class]}")
    print(f"Confidence: {confidence:.4f}")


# Demo predictions
class_keys = list(classes.keys())
image_idx = 2
for key in class_keys:
    make_predictions(key, image_idx)


# Create inference function for deployment
def predict_disease(
    image_path,
    model_path="/home/aidan/code/python/modeling/finals/rice_disease_detector_model.pth",
):
    """
    Make prediction on a rice leaf image.

    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the saved model

    Returns:
        dict: Prediction results including class name and confidence
    """
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiceLeafCNN(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess image
    image = load_and_resize_image(image_path)

    # Make prediction
    pred_class, confidence = predict_single_image(model, image, device)

    return {"disease": inverted_classes[pred_class], "confidence": confidence}


# Function to handle batch predictions
def predict_disease_batch(
    image_paths,
    model_path="/home/aidan/code/python/modeling/finals/rice_disease_detector_model.pth",
):
    """
    Make predictions on multiple rice leaf images.

    Args:
        image_paths (list): List of paths to image files
        model_path (str): Path to the saved model

    Returns:
        list: List of prediction results
    """
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiceLeafCNN(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    results = []

    for path in image_paths:
        # Load and preprocess image
        image = load_and_resize_image(path)

        # Make prediction
        pred_class, confidence = predict_single_image(model, image, device)

        results.append(
            {
                "image_path": path,
                "disease": inverted_classes[pred_class],
                "confidence": confidence,
            }
        )

    return results


# Example usage:
# single_result = predict_disease("path/to/image.jpg")
# batch_results = predict_disease_batch(["path/to/image1.jpg", "path/to/image2.jpg"])
