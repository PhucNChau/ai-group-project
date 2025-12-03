# TensorFlow and tf.keras
import tensorflow as tf
import matplotlib.pyplot as plt

# Helper libraries
import numpy as np
from tensorflow.keras import datasets
from models import create_cnn_1, train_model
from list_optimizers import list_optimizers

for name, optimizer in list_optimizers.items():
    print(f"Training with {name} optimizer...")

    # Create model
    model = create_cnn_1((32,32,3))

    # name = 'Adam'
    result = train_model(datasets.cifar10, model, list_optimizers[name])
    
    # Draw figure
    # This section is temporary will create a func for this
    plt.figure(figsize=(12, 6))
    # for name, res in results.items():
    plt.plot(result['history']['loss'], label=f"{name}")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


