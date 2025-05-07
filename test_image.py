import torch
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


# Define the same model architecture
class RiceLeafCNN(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(RiceLeafCNN, self).__init__()
        # Convolutional layers
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.5),
        )

        # Fully connected layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 32 * 32, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Define how to preprocess the image
def load_and_preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    # Resize to 128x128
    image = cv2.resize(image, (128, 128))
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply transforms
    image_tensor = transform(image)

    return image, image_tensor


# Path to your specific image
image_path = "/home/aidan/.cache/kagglehub/datasets/nirmalsankalana/rice-leaf-disease-image/versions/1/Brownspot/BROWNSPOT1_006.jpg"

# Path to your saved model
model_path = "/home/aidan/code/python/modeling/finals/rice_disease_detector_model.pth"

# Disease class names
class_names = {0: "Bacterialblight", 1: "Blast", 2: "Brownspot", 3: "Tungro"}

# Load the model
device = torch.device("cpu")  # Use CPU
model = RiceLeafCNN(num_classes=4)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set to evaluation mode

# Load and preprocess the image
original_image, image_tensor = load_and_preprocess_image(image_path)

# Make prediction
with torch.no_grad():
    input_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, prediction = torch.max(probabilities, 1)

# Display results
predicted_class = class_names[prediction.item()]
confidence_value = confidence.item()

print(f"Predicted disease: {predicted_class}")
print(f"Confidence: {confidence_value:.4f}")

# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(original_image)
plt.title(f"Prediction: {predicted_class} ({confidence_value:.4f})")
plt.axis("off")
plt.show()
