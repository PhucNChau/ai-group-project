# Define multiple models for various datasets
from keras import layers, models, optimizers
from keras.regularizers import l2
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from custom_optimizers import CustomAdam
import numpy as np

# CNN_1 has 3 convolutional and 2 FC layers
def create_cnn_1(input_shape):
    model = models.Sequential()
    # Input layer
    model.add(layers.InputLayer(shape=input_shape))
    # Convolutional layer 32
    model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', kernel_regularizer=l2(0.0001)))
    # Max pooling layer
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    # Convolutional layer 64
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', kernel_regularizer=l2(0.0001)))
    # Max pooling layer
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    # Convolutional layer 128
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', kernel_regularizer=l2(0.0001)))
    # Max pooling layer
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    # Need to flatten before Dense (or output layer) cause dense only allow 1d
    model.add(layers.Flatten())
    # FC 128
    model.add(layers.Dense(units=128, activation='relu'))
    # Dropout 0.5                                        
    model.add(layers.Dropout(rate=0.5))
    # FC softmax - output layer
    model.add(layers.Dense(units=10, activation='softmax'))

    return model


# CAE_1
def create_cae_1(input_shape):
    model = models.Sequential()
    # Input layer
    model.add(layers.InputLayer(shape=input_shape))

    ## Encoder layers
    # Convolutional layer 32
    model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
    # Max pooling layer
    model.add(layers.MaxPool2D(pool_size=(2,2), padding='same'))
    # Convolutional layer 8
    model.add(layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
    # Max pooling layer
    model.add(layers.MaxPool2D(pool_size=(2,2), padding='same'))

    ## Decoder layers
    # tranpose convolutional layer 8
    model.add(layers.Conv2DTranspose(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
    # Upsampling layer
    model.add(layers.UpSampling2D(size=(2,2)))
    # tranpose convolutional layer 32
    model.add(layers.Conv2DTranspose(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
    # Upsampling layer
    model.add(layers.UpSampling2D(size=(2,2)))

    # Output layer
    model.add(layers.Conv2D(filters=input_shape[2], kernel_size=(3,3), activation='tanh', padding='same'))

    return model

# Training CNN model
def train_cnn_model(dataset, model, is_colored_image):
    # Init param for model training
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    # Reduce size
    train_size = int(0.10 * train_images.shape[0])
    test_size = int(0.10 * test_images.shape[0])
    np.random.seed(42)

    train_idx = np.random.choice(len(train_images), train_size, replace=False)
    test_idx = np.random.choice(len(test_images), test_size, replace=False)

    train_images = train_images[train_idx]
    train_labels = train_labels[train_idx]

    test_images = test_images[test_idx]
    test_labels = test_labels[test_idx]
    
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    batch_size = 128
    # Epochs should be 50. Set low for quick debug
    no_of_epochs = 5

    # Transform label size to use categorical_crossentropy loss func
    train_labels = to_categorical(train_labels, num_classes=10)
    test_labels = to_categorical(test_labels, num_classes=10)
    loss_func = 'categorical_crossentropy'
    metrics = ['accuracy']
    
    results = {}

    for name, optimizer in list_optimizers.items():
        print(f"Training with {name} optimizer...")

        # Compile model
        model.compile(optimizer=optimizer,
                    loss=loss_func,
                    metrics=metrics)
    
        # Train model
        history = model.fit(
            train_images,
            train_labels,
            epochs = no_of_epochs,
            batch_size = batch_size,
            validation_data=(test_images, test_labels))
        
        # Save the model
        # dataset_name = "cifar10" if is_colored_image else "mnist"
        # model.save(f"cnn_model_{dataset_name}_{name}.keras")

        # Evaluate model
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

        results[name] = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "history": history.history
        }

    return results

# Training CAE model
def train_cae_model(dataset, model, is_colored_image):
    # Init param for model training
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    # Reduce size
    train_size = int(0.10 * train_images.shape[0])
    test_size = int(0.10 * test_images.shape[0])
    np.random.seed(42) 

    train_idx = np.random.choice(len(train_images), train_size, replace=False)
    test_idx = np.random.choice(len(test_images), test_size, replace=False)

    train_images = train_images[train_idx]
    train_labels = train_labels[train_idx]

    test_images = test_images[test_idx]
    test_labels = test_labels[test_idx]

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    batch_size = 128
    # Epochs should be 50. Set low for quick debug
    no_of_epochs = 5

    loss_func = 'mean_squared_error'
    metrics = ['mse']

    results = {}
    comparisons = {}
    if (is_colored_image == 0):
        # reshape test images to visualize reconstruction test
        results["test_images"] = test_images[..., None]
    else:
        results["test_images"] = test_images

    for name, optimizer in list_optimizers.items():
        print(f"Training with {name} optimizer...")

        # Compile model
        model.compile(optimizer=optimizer,
                    loss=loss_func,
                    metrics=metrics)
    
        # Train model
        history = model.fit(
            train_images,
            train_images,
            epochs = no_of_epochs,
            batch_size = batch_size,
            validation_data=(test_images, test_images))
        
        # Save the model
        # dataset_name = "cifar10" if is_colored_image else "mnist"
        # model.save(f"cae_model_{dataset_name}_{name}.keras")

        # Evaluate model
        test_loss = model.evaluate(test_images, test_images, verbose=2)

        # Image reconstruction
        predicted_images = model.predict(results["test_images"])

        comparisons[name] = {
            "test_loss": test_loss,
            "history": history.history,
            "predicted_images": predicted_images
        }

    results["compared_results"] = comparisons

    return results

def draw_figure(results, figure_title=""):
    # Draw figure
    plt.figure(figsize=(12, 6))
    for name, res in results.items():
        plt.plot(res['history']['loss'], label=f"{name}")
    plt.title(f"Training Loss for {figure_title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def print_results(results):
    print("\n===== FINAL RESULTS =====")
    for name, res in results.items():
        if "test_acc" in res:
            text = f"{name} Accuracy: {res["test_acc"]:.4f}, Loss: {res["test_loss"]:.4f}"
        else:
            text = f"{name} Loss: {res["test_loss"][0]:.4f}"
        print(text)

def show_image_reconstruction(results, is_colored_image):
    # number of images to show
    n = 10
    # number of rows
    rows = 1 + len(results["compared_results"])
    # Draw figure
    plt.figure(figsize=(18, 6))
    title_list = ["Original Images"]

    for i in range(n):
        # Original images (top row)
        ax = plt.subplot(rows, n, i + 1)
        if (is_colored_image):
            plt.imshow(results["test_images"][i])
        else:
            plt.imshow(results["test_images"][i].squeeze(), cmap="gray")

        plt.axis("off")

    # Reconstructed images
    j = 1
    for name, res in results["compared_results"].items():
        for i in range(n):
            ax = plt.subplot(rows, n, i + 1 + (n * j))
            if (is_colored_image):
                plt.imshow(res["predicted_images"][i])
            else:
                plt.imshow(res["predicted_images"][i].squeeze(), cmap="gray")

            plt.axis("off")

        j += 1
        title_list.append(name)

    k = len(title_list) - 1
    while k >= 0:
        plt.figtext(0.02, 0.18 + 0.2 * (len(title_list) - 1 - k), title_list[k], fontsize=14, va="center")
        k -= 1

    plt.show()

list_optimizers = {
    "SGD": optimizers.SGD(
        learning_rate=0.01
    ),
    # "Adam": optimizers.Adam(
    #     learning_rate=0.001,
    #     beta_1=0.9,
    #     beta_2=0.999
    # ),
    # "AdaMax": optimizers.Adamax(
    #     learning_rate=0.001,
    #     beta_1=0.9,
    #     beta_2=0.999
    # ),
    "CustomAdam": CustomAdam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999
    )
}

