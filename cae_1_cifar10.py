# TensorFlow
from tensorflow.keras import datasets
from models import create_cae_1, train_cae_model, draw_figure

# Conduct training, testing and show results for CIFAR10 dataset

# Create model
# cifar10 data set has size 32x32 and is color
model = create_cae_1((32,32,3))
# Train model
cae_results = train_cae_model(datasets.cifar10, model)
    
# Draw figure
draw_figure(cae_results, "CAE_1 on CIFAR10")
