# TensorFlow 
from tensorflow.keras import datasets
from models import create_cae_1, train_cae_model, draw_figure

# Conduct training, testing and show results for MNIST dataset

# Create model
# mnist data set has size 28x28 and is grayscale
model = create_cae_1((28,28,1))
# Train model
cae_results = train_cae_model(datasets.mnist, model)
    
# Draw figure
draw_figure(cae_results, "CAE_1 on MNIST")
