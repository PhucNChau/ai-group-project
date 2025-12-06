# TensorFlow
from keras import datasets
from models import create_cae_1, train_cae_model, draw_figure, show_image_reconstruction, print_results

# Conduct training, testing and show results for CIFAR10 dataset

# Create model
# cifar10 data set has size 32x32 and is color
model = create_cae_1((32,32,3))
# Train model
cae_results = train_cae_model(datasets.cifar10, model, is_colored_image=1)

# Print results
print_results(cae_results["compared_results"])
    
# Draw figure
draw_figure(cae_results["compared_results"], "CAE_1 on CIFAR10")

# Show image reconstruction
show_image_reconstruction(cae_results, is_colored_image=1)
