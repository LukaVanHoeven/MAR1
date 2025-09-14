# MAR 1a

Names: Bas de Blok, Mykola Chuprynskyy, Melissa Rueca, Luka van Hoeven

Baseline 1: Rule-based

Baseline 2: Majority label

Classifier 1: Logistic regression

Classifier 2: Feed forward

## How to use:

run the python files in this given order:

parse.py

step2_dedup.py

step3_split.py

main.py

## File descriptions:

baseline_majority.py: The majority based system returns an 'inform' label for every given sentence.

baseline_rulebased.py: The rule based system uses a dictionary with the labels acting as keys in the dictionary and the values of the dictionary contain arrays off words. The input is then matched against the words in the array, the key/label of the first match it finds is then returned as output.

main.py: Trains and evaluates all models in cli. The cli allows for user interaction with all models.

ML_logreg.py: Contains training and inference functions for the Logistic regression model.

ML_sequential.py: Contains inference function for the Feed Forward model.

parse.py: Normalizes the data, outputing into 'normalized.csv'.

step2_dedup.py: Removes all duplicates from the data, outputing into 'dedup.csv'.

step3_split.py: Splits the data in 'normalized.csv' into 'train_orig.csv' and 'test_orig.csv'. Also splits 'dedup.csv' into 'train_dedup.csv', 'test_dedup.csv'. 85% of the data is in the train sets and the remaining 15% is in the test sets.

train_sequential.py: Contains training functions for the sequential model.