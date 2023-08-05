#!/usr/bin/env python
# coding: utf-8

# # Import software libraries and load the dataset #

# In[1]:


# Import required libraries.
import sys                             # Read system parameters.
import shutil
import numpy as np                     # Work with multi-dimensional arrays and matrices.
from numpy.random import seed
import matplotlib as mpl               # Create 2D charts.
import matplotlib.pyplot as plt
import sklearn                         # Perform data mining and analysis.
import tensorflow                      # Train neural networks for deep learning.
import keras                           # Provide a frontend for TensorFlow.
from keras import datasets

# Summarize software libraries used.
print('Libraries used in this project:')
print('- Python {}'.format(sys.version))
print('- NumPy {}'.format(np.__version__))
print('- Matplotlib {}'.format(mpl.__version__))
print('- scikit-learn {}'.format(sklearn.__version__))
print('- TensorFlow {}'.format(tensorflow.__version__))
print('- Keras {}\n'.format(keras.__version__))

# Load the dataset.
shutil.rmtree('/home/jovyan/.keras')
shutil.copytree('/home/jovyan/work/.keras', '/home/jovyan/.keras')
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
print('Loaded {} training records.'.format(len(X_train.data)))
print('Loaded {} test records.'.format(len(X_test.data)))

# Uncomment the following two lines to make outcomes deterministic. Supply whatever seed values you wish.
#seed(1)
#tensorflow.random.set_seed(1)


# # Get acquainted with the dataset

# In[2]:


# Show dimensions of the training and testing sets and their labels
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing data shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)


# # Visualize the data examples

# In[14]:


# Show a preview of the first 20 images
import matplotlib.pyplot as plt

# Define class names (assuming class labels are integers from 0 to 9)
class_names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

# Show a preview of the first 20 images with labels
plt.figure(figsize=(10, 5))
for i in range(20):
    plt.subplot(2, 10, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(class_names[y_train[i]], fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()


# # Prepare the data for training with Keras

# In[15]:


# Reshape arrays to add greyscale flag.

# One-hot encode the data for each label.

from keras.utils import to_categorical

# Reshape arrays to add greyscale flag
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Convert label values to one-hot encoding
y_train_onehot = to_categorical(y_train, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)

# Print the one-hot encoding data for the first image
print("One-hot encoding for the first image:")
print(y_train_onehot[0])


# # Split the datasets

# In[16]:


# Split the training and validation datasets and their labels.

from sklearn.model_selection import train_test_split

# Split the training data into training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train_onehot, test_size=0.2, random_state=42)

# Print the shapes of the training and validation sets and their labels
print("Shape of the training data:", X_train_split.shape)
print("Shape of the validation data:", X_val.shape)
print("Shape of the training labels:", y_train_split.shape)
print("Shape of the validation labels:", y_val.shape)


# # Build the CNN structure

# In[17]:


# Import the required libraries
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense

# Create the model
model = Sequential()

# Add model layers
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Print a message to confirm the structure has been built
print("CNN structure has been built successfully.")


# # Compile the model and summarize the layers

# In[18]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summarize the layers
model.summary()


# # Plot a graph of the model

# In[19]:


# Install the required library.
get_ipython().system('pip install graphviz==0.16')


# In[21]:


from keras.utils import plot_model

# Plot a graph of the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# # Train the model

# In[23]:


from keras.utils import to_categorical

# Convert the target labels to one-hot encoded format
y_train_one_hot = to_categorical(y_train, num_classes=10)

# Train the model over 1 epoch with the one-hot encoded labels
model.fit(X_train, y_train_one_hot, epochs=1)


# # Evaluate the model on the test data

# In[25]:


# Evaluate the model on the test data, showing loss and accuracy.

from keras.utils import np_utils

# One-hot encode the test labels
y_test_one_hot = np_utils.to_categorical(y_test, num_classes=10)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test_one_hot)

# Print the results
print('Test loss:', loss)
print('Test accuracy:', accuracy)


# # Make predictions on the test data

# In[26]:


# Make predictions on the test data.

# Show the first 30 examples.

# Make predictions on the test data
predictions = model.predict(X_test)

# Show the first 30 examples
for i in range(30):
    print("Actual value:", y_test[i])
    print("Predicted value:", np.argmax(predictions[i]))
    print()


# # Visualize the predictions for 30 examples

# In[27]:


# Using the test set, show the first 30 predictions, highlighting any incorrect predictions in color.

import matplotlib.pyplot as plt

# Using the test set, show the first 30 predictions
plt.figure(figsize=(15, 15))
for i in range(30):
    plt.subplot(5, 6, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.title(f"True: {true_label}\nPredicted: {predicted_label}", color=color)
    plt.axis('off')
plt.show()


# In[ ]:




