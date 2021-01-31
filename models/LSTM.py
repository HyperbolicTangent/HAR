from tensorflow import keras
from tensorflow.keras import layers
import gin


@gin.configurable
def LSTM(rnn_units1, rnn_units2, dense_units, dropout_rate):
    inputs = layers.Input(shape=(250, 6))
    x = layers.Bidirectional(layers.LSTM(rnn_units1, return_sequences=True))(inputs)
    x = layers.Bidirectional(layers.LSTM(rnn_units2))(x)
    x = layers.Dense(dense_units)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(13, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model