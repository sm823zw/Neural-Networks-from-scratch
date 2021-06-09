import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from nn import Layer, NeuralNet

# Store the appropriate data paths
path = 'emotion_classification/'
train_path = os.path.join(path, 'train/')
test_path = os.path.join(path, 'test/')

train_files = os.listdir(train_path)
test_files = os.listdir(test_path)

train_y = np.array([1 if 'happy' in f_name else 0 for f_name in train_files])
test_y = np.array([1 if 'happy' in f_name else 0 for f_name in test_files])

# Read the data
train_x = np.stack([np.array(Image.open(train_path + f_name)).reshape(-1) for f_name in train_files])
test_x = np.stack([np.array(Image.open(test_path + f_name)).reshape(-1) for f_name in test_files])

# Center the data
mean = train_x.mean(axis=0)
train_x = train_x - mean
test_x = test_x - mean

# Perform PCA
U, S, Vt = np.linalg.svd(train_x, full_matrices=True)
components = Vt.T
K = 12
X_train = np.matmul(train_x, components[: , :K])
X_test = np.matmul(test_x, components[: , :K])
y_train = train_y.reshape((train_y.shape[0], 1))
y_test = test_y.reshape((test_y.shape[0], 1))

# Convert the labels to one hot vector representation
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train).toarray()
y_test = ohe.fit_transform(y_test).toarray()

# Scale the data (to prevent explosion or formation of NaN during ReLU)
maxx = np.abs(X_train).max(axis=0)
X_train = X_train/maxx
X_test = X_test/maxx

# Create a neural network
nn = NeuralNet()
nn.add_layer(Layer(12, 15, activation_fun='relu'))
nn.add_layer(Layer(15, 15, activation_fun='relu'))
nn.add_layer(Layer(15, 2, activation_fun='softmax'))

# Train the neural network using the following hyper-parameters
learning_rate = 0.001
momentum = 0.9
epochs = 20
errors = nn.train(X_train, y_train, learning_rate, momentum, epochs)

# Plot the loss vs epochs graph
plt.plot(errors, c = 'b', label = 'Training loss')
plt.title('Cross Entropy Loss vs Epochs with lr = 0.001 and momentum = 0.9')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.xticks([i for i in range(0, 20)])
plt.legend()
plt.show()

# Calculate Training Accuracy
print(nn.calc_accuracy(X_train, y_train))

# Calculate Test Accuracy
print(nn.calc_accuracy(X_test, y_test))
