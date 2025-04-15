import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Draw live chart of accuracy of neural network
from livelossplot import PlotLossesKeras


# fix unresolved errors
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import gradient_descent_v2

# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Target variable needs one-hot encoding to ensure each example has a probability of 1.0 for its actual class and 0.0 for others.
# Use Keras to_categorical() function for achieving this.

# from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils.np_utils import to_categorical

nirmalsankalana_rice_leaf_disease_image_path = kagglehub.dataset_download(
    "nirmalsankalana/rice-leaf-disease-image"
)


print("Data source import complete.")


# Define functions for loading and resizing images
def load_and_resize_image(file_path, target_shape=(128, 128)):
    image = cv2.imread(file_path)
    resized_image = cv2.resize(image, target_shape)
    return resized_image


# Define the function to load each image class (target) stored by individual directory.
# Each class directory containing their respective images
def load_image_class_by_directory(image_dir):
    # Load and resize images
    image_files = os.listdir(image_dir)
    images = []
    for file in image_files:
        if file.endswith(".jpg") or file.endswith(
            ".JPG"
        ):  # Assuming images are in jpg or JPG format
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
    # sec
    # plt.show()


# default install loc of kagglehub
# ~/.cache/kagglehub/datasets/nirmalsankalana/rice-leaf-disease-image/versions/1/Bacterialblight

# Define the directory containing images
# image_dir = "/kaggle/input/rice-leaf-disease-image/Bacterialblight"
image_dir = "/home/aidan/.cache/kagglehub/datasets/nirmalsankalana/rice-leaf-disease-image/versions/1/Bacterialblight"
images_Bacterialblight = load_image_class_by_directory(image_dir)
display_images(images_Bacterialblight)

# Define the directory containing images
# image_dir = "/kaggle/input/rice-leaf-disease-image/Blast"
image_dir = "/home/aidan/.cache/kagglehub/datasets/nirmalsankalana/rice-leaf-disease-image/versions/1/Blast/"
images_Blast = load_image_class_by_directory(image_dir)
display_images(images_Blast)

# Define the directory containing images
# image_dir = "/kaggle/input/rice-leaf-disease-image/Brownspot"
image_dir = "/home/aidan/.cache/kagglehub/datasets/nirmalsankalana/rice-leaf-disease-image/versions/1/Brownspot/"
images_Brownspot = load_image_class_by_directory(image_dir)
display_images(images_Brownspot)

# Define the directory containing images
# image_dir = "/kaggle/input/rice-leaf-disease-image/Tungro"
image_dir = "/home/aidan/.cache/kagglehub/datasets/nirmalsankalana/rice-leaf-disease-image/versions/1/Tungro"
images_Tungro = load_image_class_by_directory(image_dir)
display_images(images_Tungro)


# Define class labels
classes = {"Bacterialblight": 0, "Blast": 1, "Brownspot": 2, "Tungro": 3}
inverted_classes = {0: "Bacterialblight", 1: "Blast", 2: "Brownspot", 3: "Tungro"}

images_lst_lst = [images_Bacterialblight, images_Blast, images_Brownspot, images_Tungro]
# Dictionary to store the number of image samples
classes_dict = {}
for i, images in enumerate(images_lst_lst):
    classes_dict.update({inverted_classes[i]: len(images)})
    print(f"Disease: {inverted_classes[i]} --- Images: {len(images)}")

# Now, plot the classes
# plt.bar(*zip(*classes_dict.items()))
# plt.show()

# Pick the number of images to set aside as **test** set. The algorithm should not see this data as it will be used for later evaluations.
# Add a class label for each image. This is done by first flattening the image from 2D to 1D, and then appending the class number to it (such as 0 for Blight, etc.)


# Function to flatten the RGB values from 2D to 1D, returns a numpy array
def flatten_images(images):
    data_flattened = []
    for image in images:
        flattened_image = image.reshape(-1)  # Flatten the image
        data_flattened.append(flattened_image)

    print(f"Num of images: {len(data_flattened)}")
    print(f"Single image shape after flattening: {data_flattened[0].shape}")

    # Convert data to numpy array for further processing
    data_flattened = np.array(data_flattened)
    return data_flattened


# Function to assign class labels: returns a numpy array
def assign_image_class_label(images, class_label=int):
    data_labeled = []
    # Flatten the images
    data_flattened = flatten_images(images)

    for image in data_flattened:
        # Assign class label
        data_labeled.append(np.concatenate([image, [class_label]]))

    print(f"Num of images: {len(data_labeled)}")
    print(
        f"Single data shape with label: {data_labeled[0].shape} --- Class label: {class_label}\n"
    )

    # Convert data to numpy array for further processing
    data_labeled = np.array(data_labeled)
    return data_labeled


# Function to concatenate the arrays into a pandas dataframe, horizontally
def concat_arrays_to_dataframe(arrays=[]):
    # Combine to a single dataframe, vertically
    dataset = np.concatenate(arrays, axis=0)

    # Number of pixel columns, excluding the last label column
    num_pix = dataset.shape[1] - 1

    # Modify the column names
    col_lst = [f"pixel{col}" for col in range(num_pix)]
    # Append the name of the last column as label
    col_lst.append("label")

    # Convert to a dataframe and add column names
    df_dataset = pd.DataFrame(dataset, columns=col_lst)

    return df_dataset


# Split the image files into train - test set.
def split_train_test_files(images_lst_lst=[], num_test_set=int):
    train_images_lst_lst = []
    test_images_lst_lst = []
    # Iterate through the first class of images
    for images in images_lst_lst:
        train_set = images[num_test_set:]
        test_set = images[:num_test_set]

        train_images_lst_lst.append(train_set)
        test_images_lst_lst.append(test_set)

    return train_images_lst_lst, test_images_lst_lst


# Number of images to set aside as test set per class
num_test_set = 20

# Split the image files into train - test set.
train_images, test_images = split_train_test_files(images_lst_lst, num_test_set)

images_lst_array = []
# Iterate through the classes, the class index i will represent the class name/label
for i, images in enumerate(train_images):
    # Assign label to each of the images
    labeled = assign_image_class_label(images, i)
    images_lst_array.append(labeled)

# Concatenate arrays to dataframe
df_images = concat_arrays_to_dataframe(images_lst_array)
df_images.head()

"""# ✂️ Train Val Split"""


X_images = df_images.drop("label", axis=1)
y_images = df_images["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X_images, y_images, test_size=0.25, random_state=2, shuffle=True
)
print("Shape of train X:", X_train.shape)
print("Shape of train Y:", y_train.shape)
print("Shape of val X:", X_val.shape)
print("Shape of val Y:", y_val.shape)

# Display some images before scaling
X_train_RGB = np.array(X_train).reshape(-1, 128, 128, 3)
display_images(X_train_RGB)

# Display some images before scaling
X_val_RGB = np.array(X_val).reshape(-1, 128, 128, 3)
display_images(X_val_RGB)

"""# 🆗 Normalizing Dataset"""


# The pixel values ranges from 0 to 255. The MinMaxScaler makes it from 0 to 1.
# This reduces the magnitude sensitivity of the activation function for the choosen ML algorithm.
scaler = MinMaxScaler(feature_range=(0, 1))

# Convert to numpy array to remove feature names before fitting
scaler = scaler.fit(np.array(X_train))
X_train_np = scaler.transform(np.array(X_train))
X_val_np = scaler.transform(np.array(X_val))

# Reshape to RGB format
X_train_RGB = np.array(X_train_np).reshape(-1, 128, 128, 3)
X_val_RGB = np.array(X_val_np).reshape(-1, 128, 128, 3)

# Reshape targets
y_train = y_train.values.reshape(len(y_train), 1)
y_val = y_val.values.reshape(len(y_val), 1)

print("Shape of train X:", X_train_RGB.shape)
print("Shape of train Y:", y_train.shape)
print("Shape of val X:", X_val_RGB.shape)
print("Shape of val Y:", y_val.shape)

"""# 🏗️ Building Model"""

# # Draw live chart of accuracy of neural network
# from livelossplot import PlotLossesKeras
#
# # To avoid overfitting
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # Target variable needs one-hot encoding to ensure each example has a probability of 1.0 for its actual class and 0.0 for others.
# # Use Keras to_categorical() function for achieving this.
# from tensorflow.keras.utils import to_categorical

# Shape of a single image
input_shape = X_train_RGB[0].shape
num_train_images = len(X_train_RGB)
# Number of classes to be predicted
num_classes = 4
print(f"Single image shape: {input_shape}")
print(f"Number of train images: {num_train_images}")

"""Here comes the model architecture and its components."""

# Initialize the sequential model
model = Sequential()

# Add an input layer with the specified input shape
model.add(Input(shape=input_shape))

# Add a convolutional layer with 128 filters of size 3x3, using ReLU activation function
model.add(Conv2D(128, (3, 3), activation="relu"))

# Add a max-pooling layer with a filter size of 2x2
model.add(MaxPooling2D((2, 2)))

# Add dropout regularization to randomly omit neurons
model.add(Dropout(0.5))

# Add another convolutional layer with 64 filters of size 3x3, using ReLU activation function
model.add(Conv2D(64, (3, 3), activation="relu"))

# Add another max-pooling layer with a filter size of 2x2
model.add(MaxPooling2D((2, 2)))

# Add dropout regularization to randomly omit neurons
model.add(Dropout(0.5))

# Flatten the output of the previous layers
model.add(Flatten())

# Add a dense layer with 256 neurons and ReLU activation function
model.add(Dense(256, activation="relu"))

# Add dropout regularization to randomly omit neurons
model.add(Dropout(0.5))

# Add a dense output layer with the number of classes and softmax activation function
model.add(Dense(num_classes, activation="softmax"))

# Compile the model with categorical crossentropy loss and Stochastic Gradient Descent optimizer
opt = gradient_descent_v2.SGD(learning_rate=0.0001, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Display a summary of the model architecture
model.summary()

# Initialize Image Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10,  # rotation
    width_shift_range=0.1,  # horizontal shift
    height_shift_range=0.1,
    zoom_range=0.1,
)  # zoom

# Initialize regularization parameters - how many epochs to wait before training stops, if there is no further improvement
monitor_val_loss = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-3,
    patience=20,  # Wait 5 more epochs
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

# Target variable needs one-hot encoding to ensure each example has a probability of 1.0 for its actual class and 0.0 for others.
# Use Keras to_categorical() function for achieving this.

epochs = 100
batch_size = 64

# Using the flow method to augment the image on the fly.
training_data = train_datagen.flow(
    X_train_RGB, to_categorical(y_train), batch_size=batch_size
)
# Model evaluation
validation_data = (X_val_RGB, to_categorical(y_val))

history = model.fit(
    training_data,
    epochs=epochs,
    steps_per_epoch=num_train_images // batch_size,  # Number of iterations per epoch
    batch_size=batch_size,
    validation_data=validation_data,
    callbacks=[PlotLossesKeras(), monitor_val_loss],  # live chart
    verbose=1,
)

# Save model
# model.save("/kaggle/working/rice_disease_detector_model.keras")
model.save("/home/aidan/code/python/modeling/finals/rice_disease_detector_model.keras ")
# /home/aidan/code/python/modeling/finals/rice_disease_detector_model.keras

"""# 🧐 Evaluating Model
 Recall that first n images haven't been seen by the algorithm.
"""


# Function to scale and reshape the images for each class
def scale_and_reshape_images(flattened_images_lst=[]):
    images_scaled_RGB_lst = []
    for images in flattened_images_lst:
        # Scale images using same scaller used for the train and val set
        images_scaled = scaler.transform(images)
        # Reshape to RGB format
        images_scaled_RGB = np.array(images_scaled).reshape(-1, 128, 128, 3)

        images_scaled_RGB_lst.append(images_scaled_RGB)
    return images_scaled_RGB_lst


# Display a single image
def display_image(image_single):
    image_flat = image_single.reshape(1, -1)  # Flatten image
    image_inv = scaler.inverse_transform(image_flat)  # Inverse transform image
    image_reshaped = image_inv.reshape(128, 128, 3)

    image = image_reshaped.astype(np.uint8)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # commenting out for a sec for faster running
    # plt.show()


# Interprete model prediction: returns the predicted class and the confidence level in %
def interpret_model_prediction(predictions):
    # Convert inferences to list and
    pred_lst = predictions[0, :].tolist()
    # Get the max value
    max_proba = max(pred_lst)
    # Get the index position of the max_probability
    pred_idx = pred_lst.index(max_proba)

    return pred_idx, max_proba


# Define the function to make the inferences/predictions.
# Takes the actual value (String) and the image index (int from 0 to n, where n is the num_test_set)
def make_predictions(scaled_RGB_lst, image_class="", image_idx=int):
    # Get the numerical value of the class form the defined classes dictionary
    class_val = classes[image_class]

    # Get the single image
    image_single = scaled_RGB_lst[class_val][image_idx]

    # Make prediction for one image. Has to be reshaped
    pred = model.predict(image_single.reshape(1, 128, 128, 3))

    # Interpret model predictions
    pred_class, confidence = interpret_model_prediction(pred)

    # Display image
    display_image(image_single)

    print(f"Actual: {image_class}")
    print(f"Predicted: {list(classes.keys())[pred_class]}")
    print(f"Confidence: {round(confidence, 4)}")


# Flatten the images
images_lst_array = []
for images in test_images:
    # Function returns two values
    flattened = flatten_images(
        images
    )  # num_test_set is the number of images set aside as test set

    images_lst_array.append(flattened)

# Normalize and reshape the images for each class
images_scaled_RGB_lst = scale_and_reshape_images(images_lst_array)

# Keys from the classes dictionary defined above
class_keys = list(classes.keys())
# Image index from the test set
image_idx = 2
for key in class_keys:
    make_predictions(images_scaled_RGB_lst, key, image_idx)

"""# 🚀 Deploying Model
In order to be able to deploy our model, we want to consider two scenarios:
1. The user sends a single image for inference.
2. The user sends multiple images for inference.

We want to be able to handle both scenarios in the function(s) that does the preprocessing of the image(s).
"""
