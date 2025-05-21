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
import torchvision.models as models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import random
from PIL import Image, ImageEnhance, ImageFilter

# from torchvision.utils import make_grid
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("Data source import complete.")


# Define functions for loading and resizing images
def load_and_resize_image(
    file_path, target_shape=(224, 224)
):  # ResNet expects 224x224 images
    image = cv2.imread(file_path)
    if image is None:
        print(f"Warning: Could not load image at {file_path}")
        return np.zeros((target_shape[0], target_shape[1], 3), dtype=np.uint8)
    resized_image = cv2.resize(image, target_shape)
    return resized_image


# Define the function to load each image class (target) stored by individual directory
def load_image_class_by_directory(image_dir):
    # Load and resize images
    if not os.path.exists(image_dir):
        print(f"ERROR: Directory {image_dir} not found!")
        return []

    image_files = os.listdir(image_dir)
    images = []
    valid_extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]

    for file in image_files:
        if any(file.endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(image_dir, file)
            try:
                resized_image = load_and_resize_image(image_path)
                # Verify image has correct dimensions and channels
                if resized_image.shape == (224, 224, 3):
                    images.append(resized_image)
                else:
                    print(
                        f"Warning: Image {file} has incorrect shape {resized_image.shape}"
                    )
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

    print(f"Num of images: {len(images)}")
    if len(images) > 0:
        print(f"Single image shape before flattening: {images[0].shape}")
        # Add image statistics
        mean_brightness = np.mean([np.mean(img) for img in images])
        print(f"Mean image brightness: {mean_brightness:.2f}")
    else:
        print("No valid images found!")
    return images


# Display some images
def display_images(images, num_images_to_display=6, class_name=None):
    if len(images) < num_images_to_display:
        num_images_to_display = len(images)
        print(f"Only {num_images_to_display} images available")

    fig, axes = plt.subplots(1, num_images_to_display, figsize=(20, 5))
    fig.suptitle(
        f"Sample images from class: {class_name}" if class_name else "Sample images"
    )

    for i in range(num_images_to_display):
        # Convert the image to a supported depth (e.g., CV_8U) before color conversion
        image = images[i].astype(np.uint8)
        if num_images_to_display == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )  # Convert BGR to RGB for displaying with matplotlib
        ax.axis("off")
    plt.show()


# Download dataset
print("Downloading rice leaf disease dataset...")
nirmalsankalana_rice_leaf_disease_image_path = kagglehub.dataset_download(
    "nirmalsankalana/rice-leaf-disease-image"
)
print(f"Dataset downloaded to: {nirmalsankalana_rice_leaf_disease_image_path}")

# Load images from each class
base_path = "/home/aidan/.cache/kagglehub/datasets/nirmalsankalana/rice-leaf-disease-image/versions/1/"
print(f"Loading images from: {base_path}")

# Test if the base path exists
if not os.path.exists(base_path):
    print(f"ERROR: Base path {base_path} not found! Please check the path.")
    # Attempt to find the correct path
    print("Attempting to find the correct path...")
    potential_paths = [
        os.path.join(nirmalsankalana_rice_leaf_disease_image_path, "versions", "1"),
        nirmalsankalana_rice_leaf_disease_image_path,
    ]
    for path in potential_paths:
        if os.path.exists(path):
            base_path = path
            print(f"Found alternative path: {base_path}")
            break

print(f"Using base path: {base_path}")

images_Bacterialblight = load_image_class_by_directory(
    os.path.join(base_path, "Bacterialblight")
)
images_Blast = load_image_class_by_directory(os.path.join(base_path, "Blast"))
images_Brownspot = load_image_class_by_directory(os.path.join(base_path, "Brownspot"))
images_Tungro = load_image_class_by_directory(os.path.join(base_path, "Tungro"))
# Add the new Healthy class
images_Healthy = load_image_class_by_directory(os.path.join(base_path, "Healthy"))

# Display sample images from each class
print("\nDisplaying sample images from each class:")
for disease, images in [
    ("Bacterial Blight", images_Bacterialblight),
    ("Blast", images_Blast),
    ("Brown Spot", images_Brownspot),
    ("Tungro", images_Tungro),
    ("Healthy", images_Healthy),
]:
    if len(images) > 0:
        display_images(images, num_images_to_display=4, class_name=disease)
    else:
        print(f"No images to display for {disease}")

# Define class labels - now including Healthy as class 4
classes = {"Bacterialblight": 0, "Blast": 1, "Brownspot": 2, "Tungro": 3, "Healthy": 4}
inverted_classes = {
    0: "Bacterialblight",
    1: "Blast",
    2: "Brownspot",
    3: "Tungro",
    4: "Healthy",
}

# Update the list of image lists to include Healthy
images_lst_lst = [
    images_Bacterialblight,
    images_Blast,
    images_Brownspot,
    images_Tungro,
    images_Healthy,
]
# Dictionary to store the number of image samples
classes_dict = {}
for i, images in enumerate(images_lst_lst):
    classes_dict.update({inverted_classes[i]: len(images)})
    print(f"Disease: {inverted_classes[i]} --- Images: {len(images)}")

# Check for class imbalance
min_class_size = min(len(images) for images in images_lst_lst)
max_class_size = max(len(images) for images in images_lst_lst)
print(
    f"\nClass balance check - Min: {min_class_size}, Max: {max_class_size}, Ratio: {min_class_size / max_class_size:.2f}"
)
if min_class_size / max_class_size < 0.5:
    print(
        "WARNING: Significant class imbalance detected. Consider class weighting or data augmentation."
    )

# Visualize class distribution
plt.figure(figsize=(10, 6))
plt.bar(classes_dict.keys(), classes_dict.values())
plt.title("Class Distribution")
plt.xlabel("Disease Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


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
        # Check if we have enough images
        if len(images) <= num_test_set:
            print(
                f"WARNING: Class {inverted_classes[class_idx]} has only {len(images)} images, which is <= test set size ({num_test_set})"
            )
            test_count = max(1, len(images) // 2)  # Ensure at least one test image
            train_count = len(images) - test_count
            print(
                f"Adjusting to {train_count} train and {test_count} test images for this class"
            )
            test_set = images[:test_count]
            train_set = images[test_count:]
        else:
            test_set = images[:num_test_set]
            train_set = images[num_test_set:]

        # Create labels for each image
        test_labels_set = [class_idx] * len(test_set)
        train_labels_set = [class_idx] * len(train_set)

        train_images.extend(train_set)
        train_labels.extend(train_labels_set)
        test_images.extend(test_set)
        test_labels.extend(test_labels_set)

    print(
        f"Split complete - Train images: {len(train_images)}, Test images: {len(test_images)}"
    )
    return train_images, train_labels, test_images, test_labels


# Number of images to set aside as test set per class
num_test_set = 20

# Split the image files into train and test sets
train_images, train_labels, test_images, test_labels = split_train_test_files(
    images_lst_lst, num_test_set
)

# Verify class distribution in train and test sets
train_label_counts = {}
test_label_counts = {}
for i in range(len(inverted_classes)):
    train_label_counts[inverted_classes[i]] = train_labels.count(i)
    test_label_counts[inverted_classes[i]] = test_labels.count(i)

print("\nTrain set class distribution:")
for cls, count in train_label_counts.items():
    print(f"{cls}: {count} images")

print("\nTest set class distribution:")
for cls, count in test_label_counts.items():
    print(f"{cls}: {count} images")

# Define transformations for training data (including augmentation)
train_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # ResNet expects 224x224
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Define transformations for validation and test data
val_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # ResNet expects 224x224
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
batch_size = 32  # Smaller batch size for ResNet which is more memory intensive
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Print dataset and dataloader information
print(f"\nDataset sizes:")
print(f"Train: {len(train_dataset)} samples")
print(f"Validation: {len(val_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")
print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Test DataLoader to ensure it's working properly
print("\nTesting DataLoader functionality...")
try:
    # Get a batch from the training loader
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:5].tolist()}")
    print(
        f"Data type: {images.dtype}, Value range: [{images.min():.4f}, {images.max():.4f}]"
    )

    # Visualize a few processed training images
    plt.figure(figsize=(15, 6))
    for i in range(min(5, images.size(0))):
        plt.subplot(1, 5, i + 1)
        # Denormalize the image
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Class: {inverted_classes[labels[i].item()]}")
        plt.axis("off")
    plt.suptitle("Processed Training Images (after transforms)")
    plt.tight_layout()
    plt.show()

    print("DataLoader test successful!")
except Exception as e:
    print(f"DataLoader test failed: {str(e)}")


# Define ResNet50 model with transfer learning
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):  # Updated to 5 classes
        super(ResNet50Model, self).__init__()
        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=pretrained)

        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_features, num_classes)
        )
        print(f"Model initialized with {num_classes} output classes")
        print(f"Final layer input features: {in_features}")

    def forward(self, x):
        return self.resnet(x)

    # Method for freezing/unfreezing layers
    def freeze_backbone(self):
        # Freeze all parameters in the base model
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the final layers
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Always unfreeze the final fc layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        # Count trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(
            f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})"
        )


# Function to compute accuracy
def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Computing accuracy"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels


# Training function with enhanced monitoring
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=4, device="cuda"
):
    model = model.to(device)
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # Diagnostics: check a batch of data
    print("\nDiagnostics - Checking first batch:")
    batch = next(iter(train_loader))
    inputs, labels = batch
    print(f"Input shape: {inputs.shape}, Labels shape: {labels.shape}")
    print(f"Input min: {inputs.min().item():.4f}, max: {inputs.max().item():.4f}")
    print(f"Label distribution: {torch.bincount(labels)}")

    for epoch in range(num_epochs):
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
        print(
            f"\nStarting Epoch {epoch + 1}/{num_epochs} with learning rate: {current_lr:.6f}"
        )

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_losses = []

        for i, (inputs, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Training)")
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print batch statistics occasionally
            if i % 10 == 0:
                print(
                    f"  Batch {i}/{len(train_loader)}: Loss={loss.item():.4f}, "
                    f"Accuracy={(100 * correct / total):.2f}%"
                )

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Plot batch loss for this epoch
        plt.figure(figsize=(10, 4))
        plt.plot(batch_losses)
        plt.title(f"Epoch {epoch + 1} Batch Losses")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.show()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0] * len(classes)
        class_total = [0] * len(classes)

        with torch.no_grad():
            for inputs, labels in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Validation)"
            ):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Per-class accuracy
                for c in range(len(classes)):
                    class_mask = labels == c
                    class_total[c] += class_mask.sum().item()
                    class_correct[c] += ((predicted == c) & class_mask).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Print per-class validation accuracy
        print("Per-class validation accuracy:")
        for i in range(len(classes)):
            class_acc = (
                100 * class_correct[i] / max(1, class_total[i])
            )  # Avoid division by zero
            print(
                f"  {inverted_classes[i]}: {class_acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})"
            )

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = "/home/aidan/code/python/modeling/finals/rice_disease_resnet50_model_with_healthy.pth"
            torch.save(model.state_dict(), model_save_path)
            print(
                f"Model saved to {model_save_path} with validation accuracy: {val_acc:.2f}%"
            )

    return model, history


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print(f"Memory cached: {torch.cuda.memory_cached(0) / 1024**2:.1f} MB")

# Initialize model, loss function, and optimizer
model = ResNet50Model(num_classes=5, pretrained=True)  # Updated to 5 classes

# Test forward pass with a small batch
print("\nTesting model forward pass...")
try:
    # Create a small random batch
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    model = model.to(device)
    dummy_output = model(dummy_input)
    print(f"Forward pass test successful. Output shape: {dummy_output.shape}")
    print(
        f"Output values range: [{dummy_output.min().item():.4f}, {dummy_output.max().item():.4f}]"
    )
except Exception as e:
    print(f"Forward pass test failed: {str(e)}")

# Freeze most of the layers for transfer learning
model.freeze_backbone()

# Use class weights if there's significant imbalance
if min_class_size / max_class_size < 0.5:
    print("\nUsing weighted loss due to class imbalance")
    # Calculate class weights based on train_labels
    class_counts = {}
    for i in range(len(inverted_classes)):
        class_counts[i] = train_labels.count(i)

    # Calculate weights inversely proportional to class frequency
    weights = torch.FloatTensor(
        [
            len(train_labels) / (len(classes) * count)
            for count in [class_counts[i] for i in range(len(classes))]
        ]
    ).to(device)

    print(f"Class weights: {weights}")
    criterion = nn.CrossEntropyLoss(weight=weights)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    [
        {"params": model.resnet.fc.parameters(), "lr": 0.001},
        {"params": model.resnet.layer4.parameters(), "lr": 0.0001},
    ]
)

# Train the model
print("\nBeginning model training...")
model, history = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=4, device=device
)

# Plot training history
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(history["train_loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history["train_acc"], label="Training Accuracy")
plt.plot(history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curves")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history["lr"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.yscale("log")

plt.tight_layout()
plt.show()

# Evaluate model on test set
print("\nEvaluating model on test set...")
model_path = "/home/aidan/code/python/modeling/finals/rice_disease_resnet50_model_with_healthy.pth"
model.load_state_dict(torch.load(model_path))
test_accuracy, test_preds, test_labels = compute_accuracy(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Generate and plot confusion matrix
conf_matrix = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[inverted_classes[i] for i in range(len(classes))],
    yticklabels=[inverted_classes[i] for i in range(len(classes))],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Print classification report
print("\nClassification Report:")
class_names = [inverted_classes[i] for i in range(len(classes))]
report = classification_report(test_labels, test_preds, target_names=class_names)
print(report)


# Generate some challenging test cases
def create_augmented_test_cases(test_images, test_labels, num_cases=5):
    print("\nCreating augmented test cases...")
    augmented_images = []
    original_indices = []

    # Select random test images
    selected_indices = random.sample(range(len(test_images)), num_cases)

    for idx in selected_indices:
        img = test_images[idx]
        label = test_labels[idx]

        # Convert to PIL for augmentations
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Create different augmentations
        # 1. Brightness change
        brightness_enhancer = ImageEnhance.Brightness(pil_img)
        bright_img = brightness_enhancer.enhance(1.5)  # Increase brightness
        dark_img = brightness_enhancer.enhance(0.6)  # Decrease brightness

        # 2. Blur
        blurred_img = pil_img.filter(ImageFilter.GaussianBlur(radius=2))

        # 3. Contrast change
        contrast_enhancer = ImageEnhance.Contrast(pil_img)
        high_contrast = contrast_enhancer.enhance(1.5)

        # Convert back to OpenCV format
        bright_cv = cv2.cvtColor(np.array(bright_img), cv2.COLOR_RGB2BGR)
        dark_cv = cv2.cvtColor(np.array(dark_img), cv2.COLOR_RGB2BGR)
        blur_cv = cv2.cvtColor(np.array(blurred_img), cv2.COLOR_RGB2BGR)
        contrast_cv = cv2.cvtColor(np.array(high_contrast), cv2.COLOR_RGB2BGR)

        # Add to augmented images list
        augmented_images.extend([bright_cv, dark_cv, blur_cv, contrast_cv])
        original_indices.extend([idx] * 4)

    return (
        augmented_images,
        [test_labels[i] for i in original_indices],
        original_indices,
    )


# Create augmented test cases
augmented_images, augmented_labels, original_indices = create_augmented_test_cases(
    test_images, test_labels
)


# Function to make prediction on a single image
def predict_single_image(model, image, device, return_probs=False):
    model.eval()

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = val_transforms
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    if return_probs:
        return (
            prediction.item(),
            confidence.item(),
            probabilities.squeeze().cpu().numpy(),
        )
    else:
        return prediction.item(), confidence.item()
