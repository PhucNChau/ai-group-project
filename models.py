# Define multiple models for various datasets

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

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
def train_cnn_model(dataset, model):
    # Init param for model training
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    batch_size = 128
    # Epochs should be 50. Set low for quick debug
    no_of_epochs = 2

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

        # Evaluate model
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

        results[name] = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "history": history.history
        }

    return results

# Training CAE model
def train_cae_model(dataset, model):
    # Init param for model training
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    batch_size = 128
    # Epochs should be 50. Set low for quick debug
    no_of_epochs = 2

    loss_func = 'mean_squared_error'
    metrics = ['mse']

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
            train_images,
            epochs = no_of_epochs,
            batch_size = batch_size,
            validation_data=(test_images, test_images))

        # Evaluate model
        test_loss = model.evaluate(test_images,  test_images, verbose=2)

        results[name] = {
            "test_loss": test_loss,
            "history": history.history
        }

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

list_optimizers = {
    "SGD": optimizers.SGD(
        learning_rate=0.01
    ),
    "Adam": optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999
    ),
    "AdaMax": optimizers.Adamax(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999
    )
}

