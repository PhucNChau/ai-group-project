from tensorflow.keras import optimizers

list_optimizers = {
    "SGD": lambda: optimizers.SGD(
        learning_rate=0.01
    ),
    "Adam": lambda: optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999
    ),
    "AdaMax": lambda: optimizers.Adamax(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999
    )
}