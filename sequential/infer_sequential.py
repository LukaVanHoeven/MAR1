import pickle
import numpy as np
from keras.models import load_model, Sequential
from keras.preprocessing.text import Tokenizer






sentence = "kay"




# Load tokenizer
tokenizer : Tokenizer
with open("./sequential/tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# Load label mapping
with open("./sequential/output_types.pickle", "rb") as f:
    output_types = pickle.load(f)

model: Sequential = load_model('./sequential/seqmodel.keras')
input_neuron_amount = len(tokenizer.word_index) + 1  # +1 for OUTOFVOCAB

#Tokanize the sentence using the saved tokenizer
tokenized_sentence = tokenizer.texts_to_matrix([sentence], mode='count')

#Predict the label
prediction = model.predict(tokenized_sentence)

#Get the winning index
winning_index = np.argmax(prediction)

#Find the corresponding winning label
winning_label = "" 
for label in output_types:
    if output_types[label] == winning_index:
        winning_label = label
        break
print("winning label:", winning_label)