import math
import numpy as np
from keras.datasets import mnist
from numpy.lib.arraysetops import isin
from tensorflow.keras.utils import to_categorical

from numpynet.model import Model
from numpynet.layers import _Layer, Linear
from numpynet.activations import ReLU
from numpynet.losses import MSE

def load_mnist():
  (train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()
  train_imgs = np.reshape(train_imgs, (60_000, 784)) / 255.0
  test_imgs = np.reshape(test_imgs, (10_000, 784)) / 255.0
  train_labels = to_categorical(train_labels)
  test_labels = to_categorical(test_labels)
  return (train_imgs, train_labels), (test_imgs, test_labels)

def split_training_data(train_imgs, train_labels, num_batches):
  batches = []
  batch_size = math.ceil(train_imgs.shape[0] / num_batches)
  for b in range(num_batches):
    start = b * batch_size
    end = min(start + batch_size, train_imgs.shape[0])
    batches.append((train_imgs[start:end], train_labels[start:end]))
  return batches

def initialize_model():
  model = Model(loss=MSE())
  model.add_layer(Linear(784, 512))
  model.add_layer(ReLU())
  model.add_layer(Linear(512, 256))
  model.add_layer(ReLU())
  model.add_layer(Linear(256, 10))
  return model

def copy_model(model):
  model_copy = initialize_model()
  for i in range(len(model.layers)):
    if isinstance(model.layers[i], _Layer):
      model_copy.layers[i].W = np.copy(model.layers[i].W)
      model_copy.layers[i].b = np.copy(model.layers[i].b)

  return model_copy

def average_models(model, edge_models):
  for i in range(len(model.layers)):
    if isinstance(model.layers[i], _Layer):
      sum_W = np.zeros(model.layers[i].W.shape)
      sum_b = np.zeros(model.layers[i].b.shape)

      for edge_model in edge_models:
        sum_W += edge_model.layers[i].W
        sum_b += edge_model.layers[i].b

      model.layers[i].W = sum_W / len(edge_models)
      model.layers[i].b = sum_b / len(edge_models)

  return model

def train_model(model, imgs, labels, epochs, learning_rate, batch_size):
  num_samples = imgs.shape[0]
  num_batches = math.ceil(num_samples / batch_size)

  for e in range(epochs):
    print(f"Epoch: {e + 1}/{epochs}")
    total_loss = 0
    steps = 0

    for b in range(num_batches):
      start = b * batch_size
      end = min(start + batch_size, num_samples)

      X = imgs[start:end]
      t = labels[start:end]

      y = model(X)
      loss = model.loss(y, t)
      model.backward()
      model.update_step(lr=learning_rate)
      model.zero_grad()

      total_loss += loss
      steps += 1

      print(f"{b + 1}/{num_batches} - loss: {round(total_loss / steps, 4)}", end="\r")
    print(f"{b + 1}/{num_batches} - loss: {round(total_loss / steps, 4)}")

  return model