import argparse
import numpy as np

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from functional import average_models, load_mnist, split_training_data, initialize_model, copy_model, train_model

parser = argparse.ArgumentParser(description="Federated deep learning simulator.")
parser.add_argument("--num_edge_models", help="Number of distributed edge models.", type=int, default=5)
parser.add_argument("--training_epochs", help="Number of training epochs for each edge model.", type=int, default=5)
parser.add_argument("--training_lr", help="Learning rate used when training edge models.", type=float, default=0.01)
parser.add_argument("--training_batch_size", help="Batch size used when training edge models.", type=int, default=4)

args = parser.parse_args()

print(f"\033[1m --- settings ---\033[0m\nnum_edge_models     : {args.num_edge_models}\ntraining_epochs     : {args.training_epochs}\ntraining_lr         : {args.training_lr}\ntraining_batch_size : {args.training_batch_size}")

(train_imgs, train_labels), (test_imgs, test_labels) = load_mnist()

# Split up the training data into <num_edge_models> equally sized parts
training_data = split_training_data(train_imgs, train_labels, args.num_edge_models)

# Create the initial model that will be distributed to each edge node
model = initialize_model()

# Distribute the initial model to <num_edge_models> nodes
edge_models = [copy_model(model) for _ in range(args.num_edge_models)]

# Train the edge models on their local data
for i in range(args.num_edge_models):
  print(f"\n\033[1m--- training edge model {i + 1}/{args.num_edge_models} ---\033[0m")
  edge_models[i] = train_model(
    edge_models[i], 
    training_data[i][0], training_data[i][1], 
    args.training_epochs, 
    args.training_lr, 
    args.training_batch_size)

# Update the main model by averaging the parameters of all of the trained edge models
model = average_models(model, edge_models)

# Test the updated main model
print("\n\033[1m--- testing averaged model ---\033[0m")
y = model(test_imgs, grad=False)
loss = model.loss(y, test_labels, grad=False)

accuracy = 0
for i in range(10_000):
  accuracy += np.argmax(y[i]) == np.argmax(test_labels[i])
accuracy /= 10_000

print(f"AVERAGED MODEL - loss: {round(loss, 4)} - accuracy: {round(accuracy, 4)}")
