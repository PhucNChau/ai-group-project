# TensorFlow 
from tensorflow.keras import datasets
from models import create_cae_1, train_cae_model, draw_figure, show_image_reconstruction

# Conduct training, testing and show results for MNIST dataset

# Create model
# mnist data set has size 28x28 and is grayscale
model = create_cae_1((28,28,1))
# Train model
cae_results = train_cae_model(datasets.mnist, model, is_colored_image=0)
    
# Draw figure
draw_figure(cae_results["compared_results"], "CAE_1 on MNIST")

# Show image reconstruction
show_image_reconstruction(cae_results, is_colored_image=0)
