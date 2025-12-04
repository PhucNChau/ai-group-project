# TensorFlow
from tensorflow.keras import datasets
from models import create_cnn_1, train_model, draw_figure, list_optimizers

# Conduct training, testing and show results for CIFAR10 dataset

results = {}

for name, optimizer in list_optimizers.items():
    print(f"Training with {name} optimizer...")

    # Create model
    # cifar10 data set has size 32x32 and is color
    model = create_cnn_1((32,32,3))
    # Train model
    cnn_result = train_model(datasets.cifar10, model, list_optimizers[name])

    results[name] = cnn_result
    
# Draw figure
draw_figure(results, "CNN_1 on CIFAR10")
