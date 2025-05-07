import torch
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import os
import argparse
from tkinter import (
    Tk,
    filedialog,
    Button,
    Label,
    Frame,
    StringVar,
    Entry,
    DISABLED,
    NORMAL,
)
from PIL import Image, ImageTk
import threading


# Define the ResNet50 model architecture
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet50Model, self).__init__()
        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=pretrained)

        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


# Define how to preprocess the image
def load_and_preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None

    # Resize to 224x224 for ResNet50
    image = cv2.resize(image, (224, 224))
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


# Prediction function
def predict_image(image_path, model, device):
    # Load and preprocess the image
    original_image, image_tensor = load_and_preprocess_image(image_path)

    if original_image is None:
        return None, None, None

    # Make prediction
    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

        # Get all class probabilities
        all_probs = probabilities.squeeze().cpu().numpy()

    return original_image, prediction.item(), confidence.item(), all_probs


# Command line interface
def cli_interface():
    parser = argparse.ArgumentParser(
        description="Rice Leaf Disease Classification using ResNet50"
    )
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/aidan/code/python/modeling/finals/rice_disease_resnet50_model.pth",
        help="Path to the saved model",
    )

    args = parser.parse_args()

    # Disease class names
    class_names = {0: "Bacterialblight", 1: "Blast", 2: "Brownspot", 3: "Tungro"}

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResNet50Model(num_classes=4)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()  # Set to evaluation mode

    if args.image:
        # Single image prediction
        original_image, prediction, confidence, all_probs = predict_image(
            args.image, model, device
        )

        if original_image is None:
            print(f"Failed to process image: {args.image}")
            return

        predicted_class = class_names[prediction]

        print("\n===== PREDICTION RESULTS =====")
        print(f"Predicted disease: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")

        # Print all class probabilities
        print("\nAll class probabilities:")
        for class_idx, prob in enumerate(all_probs):
            print(f"{class_names[class_idx]}: {prob:.4f}")

        # Display the image
        plt.figure(figsize=(8, 8))
        plt.imshow(original_image)
        plt.title(f"Prediction: {predicted_class} ({confidence:.4f})")
        plt.axis("off")
        plt.show()
    else:
        print("No image specified. Use --image to specify an image path.")


# GUI interface
class RiceDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rice Leaf Disease Classifier")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # Disease class names
        self.class_names = {
            0: "Bacterialblight",
            1: "Blast",
            2: "Brownspot",
            3: "Tungro",
        }

        # Model path
        self.model_path_var = StringVar(
            value="/home/aidan/code/python/modeling/finals/rice_disease_resnet50_model.pth"
        )

        # Create frames
        self.top_frame = Frame(root, padx=10, pady=10)
        self.top_frame.pack(fill="x")

        self.content_frame = Frame(root, padx=10, pady=10)
        self.content_frame.pack(fill="both", expand=True)

        self.result_frame = Frame(root, padx=10, pady=10)
        self.result_frame.pack(fill="x")

        # Model path entry
        Label(self.top_frame, text="Model Path:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        Entry(self.top_frame, textvariable=self.model_path_var, width=50).grid(
            row=0, column=1, sticky="we", padx=5, pady=5
        )
        Button(self.top_frame, text="Browse", command=self.browse_model).grid(
            row=0, column=2, sticky="e", padx=5, pady=5
        )
        Button(self.top_frame, text="Load Model", command=self.load_model).grid(
            row=0, column=3, sticky="e", padx=5, pady=5
        )

        # Image area
        self.image_label = Label(self.content_frame, text="No image selected")
        self.image_label.pack(fill="both", expand=True)

        # Control buttons
        Button(self.result_frame, text="Browse Image", command=self.browse_image).grid(
            row=0, column=0, padx=5, pady=5
        )
        self.predict_button = Button(
            self.result_frame, text="Predict", command=self.predict, state=DISABLED
        )
        self.predict_button.grid(row=0, column=1, padx=5, pady=5)

        # Results
        self.result_var = StringVar(value="No prediction yet")
        Label(
            self.result_frame,
            textvariable=self.result_var,
            wraplength=600,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        # Status bar
        self.status_var = StringVar(value="Ready")
        Label(
            root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w"
        ).pack(side="bottom", fill="x")

        # Initialize variables
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.status_var.set(f"Ready (using {self.device})")
        self.current_image_path = None

        # Try to load the default model
        self.load_model()

    def browse_model(self):
        model_path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
        )
        if model_path:
            self.model_path_var.set(model_path)

    def load_model(self):
        model_path = self.model_path_var.get()
        if not os.path.exists(model_path):
            self.status_var.set(f"Error: Model file not found at {model_path}")
            return

        try:
            # Load the model in a separate thread to keep UI responsive
            self.status_var.set("Loading model...")
            threading.Thread(target=self._load_model, args=(model_path,)).start()
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")

    def _load_model(self, model_path):
        try:
            self.model = ResNet50Model(num_classes=4)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.root.after(
                0,
                lambda: self.status_var.set(
                    f"Model loaded successfully from {model_path}"
                ),
            )
            self.root.after(0, lambda: self.predict_button.config(state=NORMAL))
        except Exception as e:
            self.root.after(
                0, lambda: self.status_var.set(f"Error loading model: {str(e)}")
            )

    def browse_image(self):
        image_path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*")],
        )
        if image_path:
            self.current_image_path = image_path
            self.display_image(image_path)
            if self.model is not None:
                self.predict_button.config(state=NORMAL)

    def display_image(self, image_path):
        try:
            # Load image with PIL for display
            img = Image.open(image_path)
            # Resize for display while maintaining aspect ratio
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)

            # Update image label
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
            self.status_var.set(f"Image loaded: {image_path}")
        except Exception as e:
            self.status_var.set(f"Error displaying image: {str(e)}")

    def predict(self):
        if not self.current_image_path or not self.model:
            return

        # Start prediction in a separate thread to keep UI responsive
        self.predict_button.config(state=DISABLED)
        self.status_var.set("Predicting...")
        threading.Thread(target=self._predict).start()

    def _predict(self):
        try:
            original_image, prediction, confidence, all_probs = predict_image(
                self.current_image_path, self.model, self.device
            )

            if original_image is None:
                self.root.after(
                    0,
                    lambda: self.status_var.set(
                        f"Failed to process image: {self.current_image_path}"
                    ),
                )
                self.root.after(0, lambda: self.predict_button.config(state=NORMAL))
                return

            predicted_class = self.class_names[prediction]

            # Prepare result text
            result_text = f"Predicted Disease: {predicted_class}\nConfidence: {confidence:.4f}\n\nAll Class Probabilities:\n"
            for class_idx, prob in enumerate(all_probs):
                result_text += f"- {self.class_names[class_idx]}: {prob:.4f}\n"

            # Update UI in the main thread
            self.root.after(0, lambda: self.result_var.set(result_text))
            self.root.after(0, lambda: self.status_var.set("Prediction complete"))
            self.root.after(0, lambda: self.predict_button.config(state=NORMAL))

        except Exception as e:
            self.root.after(
                0, lambda: self.status_var.set(f"Error during prediction: {str(e)}")
            )
            self.root.after(0, lambda: self.predict_button.config(state=NORMAL))


# Main function to decide which interface to use
def main():
    import sys

    if len(sys.argv) > 1:
        # Command line mode
        cli_interface()
    else:
        # GUI mode
        root = Tk()
        app = RiceDiseaseApp(root)
        root.mainloop()


if __name__ == "__main__":
    main()
