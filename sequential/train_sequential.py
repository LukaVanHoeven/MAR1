import keras
import matplotlib.pyplot as plt
from keras import layers
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import pickle 
# We look at the entire dataset to generate a "vocabulary"
# Each word in the vocabulary gets a neuron in the input layer
# One neuron is left for "OUTOFVOCAB" (words outside vocabulary)

with open('dialog_acts.dat', 'r') as file:
    data = file.readlines()
    
cleaned_data = []
for index, line in enumerate(data):
    [text, label] = line.split(maxsplit=1)
    cleaned_data.append((text, label))


labels = []
sentences = []
for i in range(len(cleaned_data)):
    label = cleaned_data[i][0]
    sentences.append(cleaned_data[i][1])
    labels.append(label)

#We make a tokenizer, which will convert words into numbers 
#It also adds the OUTOFVOCAB token for words outside the vocabulary
tokenizer = Tokenizer(oov_token="OUTOFVOCAB")
tokenizer.fit_on_texts(sentences)



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

#We save the tokenizer/the output types for later use in inference
with open('./sequential/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)

with open('./sequential/output_types.pickle', 'wb') as handle:
    pickle.dump(output_types, handle)
    
#We convert the sentences into the bag of words format
#This is a vector of size vocabulary+1, where each position indicates how many times the word was used in the sentence
x = tokenizer.texts_to_matrix(sentences)
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

model.compile(metrics=["accuracy"],
              optimizer="adam",
              loss="categorical_crossentropy"
              )

history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

model.save('./sequential/seqmodel.keras')


#make a confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(18,5))
y_prediction = model.predict(X_test)
cm = confusion_matrix(y_test.argmax(axis=1), y_prediction.argmax(axis=1), normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=output_types.keys())
disp.plot(ax=axes[0])
axes[0].tick_params(axis='x', rotation=45)
#plot loss
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Test Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training and Test Loss')
axes[1].legend()

# make a plot of the accuracy
axes[2].plot(history.history['accuracy'], label='Training Accuracy')
axes[2].plot(history.history['val_accuracy'], label='Test Accuracy')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy')
axes[2].set_title('Training and Test Accuracy')
axes[2].legend()

plt.show()

    

