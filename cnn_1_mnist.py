# TensorFlow 
from keras import datasets
from models import create_cnn_1, train_cnn_model, draw_figure, print_results

# Conduct training, testing and show results for MNIST dataset

# Create model
# mnist data set has size 28x28 and is grayscale
model = create_cnn_1((28,28,1))
# Train model
cnn_results = train_cnn_model(datasets.mnist, model, is_colored_image=0)

# Print results
print_results(cnn_results)

# Draw figure
draw_figure(cnn_results, "CNN_1 on MNIST")
