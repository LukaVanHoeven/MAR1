import keras
from keras import layers
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np
from keras.models import load_model


#There are 15 output types, for clarification sake we make a dictionary so we can map them properly
output_types = {
    "ack": 0,
    "affirm": 1,
    "bye": 2,
    "confirm": 3,
    "deny": 4,
    "hello": 5,
    "inform": 6,
    "negate": 7,
    "null": 8,
    "repeat": 9,
    "reqalts": 10,
    "reqmore": 11,
    "request": 12,
    "restart": 13,
    "thankyou": 14
}

outputs = [
    "ack", 
    "affirm", 
    "bye", 
    "confirm", 
    "deny", 
    "hello",  
    "inform", 
    "negate", 
    "null", 
    "repeat", 
    "reqalts", 
    "reqmore", 
    "request", 
    "restart", 
    "thankyou"
]

def train_sequential(model_name: str, df: pd.DataFrame)-> tuple[str,str]:
    """
    Train the sequential model.

    @param model_name (str): 
    @param df (pd.Dataframe):

    @return tuple[str,str]: A tuple containing:
        - The file path containing the sequential model.
        - The file path containing the corresponding tokenizer. 
    """
    tokenizer_name = f'{model_name}.pickle'
    sequential_name = f'{model_name}.keras'

    labels = df["label"].tolist()
    sentences = df["text"].tolist()

    #We make a tokenizer, which will convert words into numbers 
    #It also adds the OUTOFVOCAB token for words outside the vocabulary
    tokenizer = Tokenizer(oov_token="OUTOFVOCAB")
    tokenizer.fit_on_texts(sentences)

    #We save the tokenizer/the output types for later use in inference
    with open(tokenizer_name, 'wb') as handle:
        pickle.dump(tokenizer, handle)

    #We convert the sentences into the bag of words format
    #This is a vector of size vocabulary+1, where each position indicates how many times the word was used in the sentence
    x = tokenizer.texts_to_matrix(sentences, mode='count')
    y = to_categorical([output_types[label] for label in labels], num_classes=15)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    #The input neuron amount is the size of the vocabulary
    #For the input, we can input the bag of words, so how many times the words are used in the input sentence
    input_neuron_amount = len(tokenizer.word_index)

    #We create a simple feedforward neural network
    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(input_neuron_amount + 1,)), #We do +1 because of the OUTOFVOCAB token
            layers.Dense(128, activation="relu", name="layer2"),
            layers.Dense(64, activation="relu", name="layer3"),
            layers.Dense(32, activation="relu", name="layer4"),
            layers.Dense(15, name="output", activation="softmax"),
        ]
    )

    model.compile(
        metrics=["accuracy"],
        optimizer="adam",
        loss="categorical_crossentropy"
    )

    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    model.save(sequential_name)

    return sequential_name, tokenizer_name


def sequential(data: list[str], model: str, tokenizer: str)-> list[str]:
    """
    Infers the sequential model.

    @param data (list[str]): List of all the utterances that need to be
        classified.
    @param model (str): The file path containing the sequential model.
    @param tokenizer (str): The file path containing the corresponding
        tokenizer. 

    @return list[str]: List containing all the classified labels.
    """
    with open(tokenizer, "rb") as f:
        tokenizer = pickle.load(f)

    model = load_model(model)

    tokenized_sentences = tokenizer.texts_to_matrix(data, mode='count')

    predictions = model.predict(tokenized_sentences, verbose=0)

    winning_indices = np.argmax(predictions, axis=1)
    return [outputs[i] for i in winning_indices]
