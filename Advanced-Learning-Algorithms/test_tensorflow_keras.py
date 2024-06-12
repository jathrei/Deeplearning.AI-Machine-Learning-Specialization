import tensorflow as tf
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy # type: ignore
from tensorflow.keras.activations import sigmoid # type: ignore

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Simple model to test TensorFlow and Keras integration
model = Sequential([
    Input(shape=(4,)),
    Dense(10, activation=sigmoid),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])

# Print model summary
model.summary()
