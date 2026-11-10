import tensorflow as tf

from src import config


def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_shallow_nn(input_dim):
    """Neural network with one hidden layer."""
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])

    return compile_model(model)


def build_wide_nn(input_dim):
    """Neural network with one wide hidden layer."""
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])

    return compile_model(model)


def build_deep_nn(input_dim):
    """Deep neural network with multiple hidden layers."""
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])

    return compile_model(model)


def build_regularized_deep_nn(input_dim):
    """Deep neural network with stronger dropout and L2 regularization."""
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    l2 = tf.keras.regularizers.l2(0.001)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=l2,
        ),
        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=l2,
        ),
        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=l2,
        ),
        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=l2,
        ),

        tf.keras.layers.Dense(
            4,
            activation="softmax",
        ),
    ])

    return compile_model(model)


def build_regularized_deep_light_nn(input_dim):
    """Deep neural network with lighter dropout and L2 regularization."""
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    l2 = tf.keras.regularizers.l2(0.0001)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=l2,
        ),
        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=l2,
        ),
        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=l2,
        ),
        tf.keras.layers.Dropout(0.10),

        tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=l2,
        ),

        tf.keras.layers.Dense(
            4,
            activation="softmax",
        ),
    ])

    return compile_model(model)