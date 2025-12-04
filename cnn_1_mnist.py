# TensorFlow 
from tensorflow.keras import datasets
from models import create_cnn_1, train_model, draw_figure, list_optimizers

# Conduct training, testing and show results for MNIST dataset

results = {}

for name, optimizer in list_optimizers.items():
    print(f"Training with {name} optimizer...")

    # Create model
    # mnist data set has size 28x28 and is grayscale
    model = create_cnn_1((28,28,1))
    # Train model
    cnn_result = train_model(datasets.mnist, model, list_optimizers[name])

    results[name] = cnn_result
    
# Draw figure
draw_figure(results, "CNN_1 on MNIST")
