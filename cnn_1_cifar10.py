# TensorFlow and tf.keras
import tensorflow as tf
import matplotlib.pyplot as plt

# Helper libraries
import numpy as np
from tensorflow.keras import datasets
from models import create_cnn_1, train_model, draw_figure, list_optimizers

results = {}

for name, optimizer in list_optimizers.items():
    print(f"Training with {name} optimizer...")

    # Create model
    model = create_cnn_1((32,32,3))
    # Train model
    cnn_result = train_model(datasets.cifar10, model, list_optimizers[name])

    results[name] = cnn_result
    
# Draw figure
draw_figure(results)

