from typing import List
from keras.preprocessing.text import Tokenizer

with open('dialog_acts.dat', 'r') as file:
    data = file.readlines()
    
cleaned_data = []
for index, line in enumerate(data):
    [text, label] = line.split(maxsplit=1)
    cleaned_data.append((text, label))

sentence = "" 
for i in range(len(cleaned_data)):
    label = cleaned_data[i][0]
    sentence = sentence + cleaned_data[i][1]

    
# Source for function: https://en.wikipedia.org/wiki/Bag-of-words_model
def get_bow(sentence: List[str]) -> None:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence)
    sequences = tokenizer.texts_to_sequences(sentence)
    word_index = tokenizer.word_index 
    bow = {}
    for key in word_index:
        bow[key] = sequences[0].count(word_index[key])
    print(type(bow))
    return bow
bow_entire_data = get_bow(sentence)





    