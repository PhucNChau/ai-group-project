from keras.models import load_model
from keras import datasets
from models import draw_figure, show_image_reconstruction
from custom_optimizers import CustomAdam
import numpy as np

# CAE - CIFAR10
dataset_name = 'cifar10'
is_colored_image = 0
(_, _), (test_images, test_labels) = datasets.cifar10.load_data()

# Reduce size
test_size = 1000
np.random.seed(42)   # any fixed number you like
test_idx = np.random.choice(len(test_images), test_size, replace=False)

test_images = test_images[test_idx]
test_labels = test_labels[test_idx]

test_images = test_images / 255.0

results = {}
comparisons = {}
if (is_colored_image == 0):
    # reshape test images to visualize reconstruction test
    results["test_images"] = test_images[..., None]
else:
    results["test_images"] = test_images

names = ["SGD", "Adam", "AdaMax", "Adam"]
trained_model_paths = {
    "SGD": f"cae_model_{dataset_name}_SGD.keras",
    "Adam": f"cae_model_{dataset_name}_Adam.keras",
    "AdaMax": f"cae_model_{dataset_name}_AdaMax.keras",
    "CustomAdam": f"cae_model_{dataset_name}_CustomAdam.keras",
}

for name in names:
    model_path = trained_model_paths[name]

    model = load_model(model_path)

    test_loss = model.evaluate(test_images, test_images, verbose=2)

    predicted_images = model.predict(results["test_images"])

    comparisons[name] = {
        "test_loss": test_loss,
        "predicted_images": predicted_images
    }

results["compared_results"] = comparisons

show_image_reconstruction(results, is_colored_image=0)
