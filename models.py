# Define multiple models for various datasets

from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

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

# CNN_2 has 5 convolutional and 3 FC layers
def create_cnn_2(input_shape):
    model = models.Sequential()



    return model

# CNN_3 has 7 convolutional and 3 FC layers
def create_cnn_3(input_shape):
    model = models.Sequential()



    return model