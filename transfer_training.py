from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.utils import to_categorical


# Define image size
IMG_SIZE = 224

# Define categories and their labels
CATEGORIES = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}

# Initialize X and y arrays
X = []
y = []
label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # keep track of the number of images added for each category
categories = list(CATEGORIES.keys())  # get a list of category names
while True:
    stop = True
    for category in categories:
        if label_counts[CATEGORIES[category]] < len(os.listdir(os.path.join("small_flower_dataset", category))):
            # Load the next image and add it to X and y
            image_path = os.path.join("small_flower_dataset", category, os.listdir(os.path.join("small_flower_dataset", category))[label_counts[CATEGORIES[category]]])
            image = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
            image_array = np.array(image)
            X.append(image_array)
            y.append(CATEGORIES[category])
            label_counts[CATEGORIES[category]] += 1
            stop = False
    if stop:
        break

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into train and test sets
split = int(len(X) * 0.9) # 80% train, 20% test
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Load the MobileNetV2 model pre-trained on ImageNet
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze all the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add your own output layer with 5 neurons and a softmax activation function
x = Flatten()(base_model.output)
x = Dense(5, activation='softmax')(x)

# Create a new model that includes both the pre-trained base model and your custom output layer
model = Model(inputs=base_model.input, outputs=x)

# Define the optimizer
optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)


# Compile the model with an appropriate optimizer, loss function, and metrics
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


y_train = to_categorical(y_train, num_classes=len(CATEGORIES))
y_test = to_categorical(y_test, num_classes=len(CATEGORIES))

# Train the model on your own dataset
model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

# Predict the first 10 results in X_test
y_pred = model.predict(X_test[50:100])
y_pred_labels = [list(CATEGORIES.keys())[list(CATEGORIES.values()).index(np.argmax(p))] for p in y_pred]

# Convert the one-hot encoded true labels to labels
y_true_labels = list(y_test[50:100])
true_labels = []
for label in y_true_labels:
    true_labels.append(list(CATEGORIES.keys())[list(CATEGORIES.values()).index(np.argmax(label))])


fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(20,10))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i+50])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"True: {true_labels[i]}\nPredicted: {y_pred_labels[i]}")
    
plt.tight_layout()
plt.show()