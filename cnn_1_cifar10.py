# TensorFlow
from tensorflow.keras import datasets
from models import create_cnn_1, train_cnn_model, draw_figure

# Conduct training, testing and show results for CIFAR10 dataset

# Create model
# cifar10 data set has size 32x32 and is color
model = create_cnn_1((32,32,3))
# Train model
cnn_results = train_cnn_model(datasets.cifar10, model)
    
# Draw figure
draw_figure(cnn_results, "CNN_1 on CIFAR10")
