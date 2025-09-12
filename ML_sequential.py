import pickle
import numpy as np
from keras.models import load_model, Sequential


output_types = [
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


def sequential(data, model, tokenizer):
    with open(tokenizer, "rb") as f:
        tokenizer = pickle.load(f)

    model: Sequential = load_model(model)

    tokenized_sentences = tokenizer.texts_to_matrix(data, mode='count')

    predictions = model.predict(tokenized_sentences)

    winning_indices = np.argmax(predictions, axis=1)
    return [output_types[i] for i in winning_indices]

