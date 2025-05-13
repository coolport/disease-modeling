import torch  # PyTorch - Deep learning framework
import cv2  # OpenCV - Computer vision library for image processing
import numpy as np  # NumPy - Numerical operations
from torchvision import transforms  # Image transformation utilities
import matplotlib.pyplot as plt  # Plotting library
import torchvision.models as models  # Pre-trained models
import torch.nn as nn  # Neural network module
import os  # Operating system interface
from tkinter import (  # GUI toolkit
    Tk,  # Root window
    filedialog,  # File selection dialogs
    Button,  # Button widgets
    Label,  # Text/image display widgets
    Frame,  # Container widgets
    StringVar,  # String variables for widgets
    Entry,  # Text entry widgets
    DISABLED,  # Constant for disabled state
    NORMAL,  # Constant for normal state
)
from PIL import Image, ImageTk  # Python Imaging Library for image handling
import threading  # Threading for background tasks


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class ResNet50Model(nn.Module):
    """
    Define a custom ResNet50-based model for rice leaf disease classification.

    This model uses transfer learning by starting with a pre-trained ResNet50
    and replacing the final fully-connected layer to classify into 4 disease classes.
    """

    def __init__(self, num_classes=4, pretrained=True):
        """
        Initialize the model architecture.

        Args:
            num_classes (int): Number of disease classes to classify (default: 4)
            pretrained (bool): Whether to use pre-trained weights from ImageNet (default: True)
        """
        super(ResNet50Model, self).__init__()

        # Load pre-trained ResNet50 model with weights trained on ImageNet
        self.resnet = models.resnet50(pretrained=pretrained)

        # Save the number of features in the last layer for reference
        in_features = self.resnet.fc.in_features

        # Replace the final fully connected layer with a custom classifier
        # The dropout layer helps prevent overfitting
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),  # 50% dropout for regularization
            nn.Linear(in_features, num_classes),  # Final layer for classification
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Raw logits for each disease class
        """
        return self.resnet(x)  # Pass input through the entire network


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================
def load_and_preprocess_image(image_path):
    """
    Load an image from disk and preprocess it for the model.

    This function handles:
    1. Loading the image from the file path
    2. Resizing to the expected input size for ResNet50 (224x224)
    3. Converting from BGR to RGB color space (OpenCV uses BGR by default)
    4. Applying normalization transformations required by ResNet50

    Args:
        image_path (str): Path to the image file

    Returns:
        tuple: (original_image, preprocessed_tensor) where:
            - original_image is the resized RGB image as a numpy array
            - preprocessed_tensor is the normalized tensor ready for the model
    """
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None

    # Resize to 224x224 pixels (ResNet50's expected input size)
    image = cv2.resize(image, (224, 224))

    # Convert from BGR (OpenCV default) to RGB (what our model expects)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define preprocessing transformations:
    # 1. Convert to PIL image (required by torchvision transforms)
    # 2. Convert to PyTorch tensor
    # 3. Normalize with ImageNet mean and std values (since we're using pretrained model)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),  # Converts to tensor and scales to [0,1]
            # Normalize using ImageNet mean and std for each channel
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply transforms to get model-ready tensor
    image_tensor = transform(image)

    return image, image_tensor


# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict_image(image_path, model, device):
    """
    Make a prediction on an image using the loaded model.

    This function:
    1. Loads and preprocesses the image
    2. Passes it through the model
    3. Gets prediction and confidence values

    Args:
        image_path (str): Path to the image file
        model (nn.Module): Loaded PyTorch model
        device (torch.device): Device to run inference on (CPU or CUDA)

    Returns:
        tuple: (original_image, prediction_class_index, confidence_score, all_probabilities)
    """
    # Load and preprocess the image
    original_image, image_tensor = load_and_preprocess_image(image_path)

    if original_image is None:
        return None, None, None, None

    # Run inference without calculating gradients (not needed for prediction)
    with torch.no_grad():
        # Add batch dimension and move to appropriate device
        input_tensor = image_tensor.unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

        # Forward pass through the model
        output = model(input_tensor)  # Raw logits

        # Convert logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(output, dim=1)

        # Get the highest probability and its index (class)
        confidence, prediction = torch.max(probabilities, 1)

        # Extract all class probabilities as numpy array
        all_probs = probabilities.squeeze().cpu().numpy()

    return original_image, prediction.item(), confidence.item(), all_probs


# =============================================================================
# GUI APPLICATION CLASS
# =============================================================================
class RiceDiseaseApp:
    """
    Tkinter-based GUI application for the Rice Leaf Disease Classifier.

    This class provides a complete graphical interface for:
    1. Loading a trained model
    2. Selecting images for classification
    3. Viewing and interpreting prediction results
    """

    def __init__(self, root):
        """
        Initialize the GUI application.

        Args:
            root (Tk): The root Tkinter window
        """
        # Set up the main window
        self.root = root
        self.root.title("Rice Leaf Disease Classifier")
        self.root.geometry("800x600")  # Initial window size
        self.root.resizable(True, True)  # Allow window resizing

        # Define disease class names (mapping from index to disease name)
        self.class_names = {
            0: "Bacterialblight",
            1: "Blast",
            2: "Brownspot",
            3: "Tungro",
        }

        # String variable to hold model path
        # Default path is provided but can be changed by user
        self.model_path_var = StringVar(
            value="/home/aidan/code/python/modeling/finals/rice_disease_resnet50_model.pth"
        )

        # =====================================================================
        # Create frame layout
        # =====================================================================

        # Top frame for model selection controls
        self.top_frame = Frame(root, padx=10, pady=10)
        self.top_frame.pack(fill="x")  # Fill horizontally

        # Middle frame for the image display
        self.content_frame = Frame(root, padx=10, pady=10)
        self.content_frame.pack(fill="both", expand=True)  # Fill all available space

        # Bottom frame for prediction results and controls
        self.result_frame = Frame(root, padx=10, pady=10)
        self.result_frame.pack(fill="x")  # Fill horizontally

        # =====================================================================
        # Top frame components (model selection)
        # =====================================================================

        # Model path label
        Label(self.top_frame, text="Model Path:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )

        # Model path entry field
        Entry(self.top_frame, textvariable=self.model_path_var, width=50).grid(
            row=0, column=1, sticky="we", padx=5, pady=5
        )

        # Browse button to select model file
        Button(self.top_frame, text="Browse", command=self.browse_model).grid(
            row=0, column=2, sticky="e", padx=5, pady=5
        )

        # Load button to load the selected model
        Button(self.top_frame, text="Load Model", command=self.load_model).grid(
            row=0, column=3, sticky="e", padx=5, pady=5
        )

        # =====================================================================
        # Content frame components (image display)
        # =====================================================================

        # Label to display the selected image
        self.image_label = Label(self.content_frame, text="No image selected")
        self.image_label.pack(fill="both", expand=True)

        # =====================================================================
        # Result frame components (controls and results display)
        # =====================================================================

        # Button to browse for an image
        Button(self.result_frame, text="Browse Image", command=self.browse_image).grid(
            row=0, column=0, padx=5, pady=5
        )

        # Button to run prediction (initially disabled until model is loaded)
        self.predict_button = Button(
            self.result_frame, text="Predict", command=self.predict, state=DISABLED
        )
        self.predict_button.grid(row=0, column=1, padx=5, pady=5)

        # String variable to hold prediction results
        self.result_var = StringVar(value="No prediction yet")

        # Label to display prediction results
        Label(
            self.result_frame,
            textvariable=self.result_var,
            wraplength=600,  # Wrap text at 600 pixels
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        # =====================================================================
        # Status bar (bottom of window)
        # =====================================================================

        # String variable to hold status messages
        self.status_var = StringVar(value="Ready")

        # Status bar label
        Label(
            root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w"
        ).pack(side="bottom", fill="x")

        # =====================================================================
        # Initialize app state variables
        # =====================================================================

        self.model = None  # Will hold the loaded PyTorch model

        # Determine if CUDA (GPU) is available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.status_var.set(f"Ready (using {self.device})")

        self.current_image_path = None  # Will hold the path to currently selected image

        # Try to load the default model when starting
        self.load_model()

    def browse_model(self):
        """
        Open a file dialog to select a model file.

        This updates the model path in the entry field but doesn't load the model yet.
        """
        model_path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
        )
        if model_path:  # If user didn't cancel the dialog
            self.model_path_var.set(model_path)

    def load_model(self):
        """
        Load the PyTorch model from the specified path.

        This method initiates the loading process in a separate thread to keep
        the UI responsive while loading the model, which could take some time.
        """
        model_path = self.model_path_var.get()

        # Check if the model file exists
        if not os.path.exists(model_path):
            self.status_var.set(f"Error: Model file not found at {model_path}")
            return

        try:
            # Update status to indicate loading has started
            self.status_var.set("Loading model...")

            # Start loading process in a separate thread to keep UI responsive
            threading.Thread(target=self._load_model, args=(model_path,)).start()
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")

    def _load_model(self, model_path):
        """
        Helper method to load the model in a background thread.

        Args:
            model_path (str): Path to the PyTorch model file
        """
        try:
            # Initialize the model architecture
            self.model = ResNet50Model(num_classes=4)

            # Load the saved weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

            # Set model to evaluation mode (disables dropout, batch norm, etc.)
            self.model.eval()

            # Update UI in main thread to show success
            # We use root.after(0, ...) to schedule UI updates on the main thread
            self.root.after(
                0,
                lambda: self.status_var.set(
                    f"Model loaded successfully from {model_path}"
                ),
            )

            # Enable the predict button now that the model is loaded
            self.root.after(0, lambda: self.predict_button.config(state=NORMAL))
        except Exception as e:
            # Update UI in main thread to show error
            self.root.after(
                0, lambda: self.status_var.set(f"Error loading model: {str(e)}")
            )

    def browse_image(self):
        """
        Open a file dialog to select an image for prediction.

        This also displays the selected image in the UI.
        """
        image_path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*")],
        )
        if image_path:  # If user didn't cancel the dialog
            self.current_image_path = image_path
            self.display_image(image_path)  # Show the image in the UI

            # Enable predict button if model is already loaded
            if self.model is not None:
                self.predict_button.config(state=NORMAL)

    def display_image(self, image_path):
        """
        Display an image in the UI.

        Args:
            image_path (str): Path to the image file
        """
        try:
            # Load image with PIL
            img = Image.open(image_path)

            # Resize for display while maintaining aspect ratio
            # thumbnail modifies the image in-place
            img.thumbnail((400, 400))

            # Convert PIL image to Tkinter-compatible format
            photo = ImageTk.PhotoImage(img)

            # Update image display
            self.image_label.config(image=photo)
            self.image_label.image = (
                photo  # Keep a reference to prevent garbage collection
            )

            # Update status bar
            self.status_var.set(f"Image loaded: {image_path}")
        except Exception as e:
            self.status_var.set(f"Error displaying image: {str(e)}")

    def predict(self):
        """
        Start the prediction process for the current image.

        This method initiates the prediction in a separate thread to keep
        the UI responsive during processing.
        """
        # Check if we have both an image and a model
        if not self.current_image_path or not self.model:
            return

        # Disable the predict button while processing
        self.predict_button.config(state=DISABLED)

        # Update status bar
        self.status_var.set("Predicting...")

        # Start prediction in a separate thread
        threading.Thread(target=self._predict).start()

    def _predict(self):
        """
        Helper method to run prediction in a background thread.

        This processes the selected image and updates the UI with results.
        """
        try:
            # Call the predict_image function defined earlier
            original_image, prediction, confidence, all_probs = predict_image(
                self.current_image_path, self.model, self.device
            )

            # Handle case where image couldn't be processed
            if original_image is None:
                self.root.after(
                    0,
                    lambda: self.status_var.set(
                        f"Failed to process image: {self.current_image_path}"
                    ),
                )
                self.root.after(0, lambda: self.predict_button.config(state=NORMAL))
                return

            # Get the disease name from the class index
            predicted_class = self.class_names[prediction]

            # Format a detailed result text showing all probabilities
            result_text = f"Predicted Disease: {predicted_class}\nConfidence: {confidence:.4f}\n\nAll Class Probabilities:\n"
            for class_idx, prob in enumerate(all_probs):
                result_text += f"- {self.class_names[class_idx]}: {prob:.4f}\n"

            # Update UI elements in the main thread
            self.root.after(0, lambda: self.result_var.set(result_text))
            self.root.after(0, lambda: self.status_var.set("Prediction complete"))
            self.root.after(0, lambda: self.predict_button.config(state=NORMAL))

        except Exception as e:
            # Handle any errors during prediction
            self.root.after(
                0, lambda: self.status_var.set(f"Error during prediction: {str(e)}")
            )
            self.root.after(0, lambda: self.predict_button.config(state=NORMAL))


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
def main():
    """
    Main entry point for the application.

    This creates the Tk root window and starts the application.
    """
    # Create the root Tkinter window
    root = Tk()

    # Initialize our application with the root window
    app = RiceDiseaseApp(root)

    # Start the Tkinter event loop
    # This will block until the window is closed
    root.mainloop()


# Run the application when this script is executed directly
if __name__ == "__main__":
    main()
