import keras
from keras import layers
from keras import ops

#There are 15 classes in the dataset

# We look at the entire dataset to generate a "vocabulary"
# Each word in the vocabulary gets a neuron in the input layer
# One neuron is left for "NULL" (words outside vocabulary)
# Then we feed it through two hidden layers



model = keras.Sequential(
    [
        layers.Dense(64, activation="relu", name="layer1"),
        layers.Dense(32, activation="relu", name="layer2"),
        layers.Dense(15, name="layer3"),
    ]
)