# MAR 1a

Names:Bas de Blok,Mykola Chuprynskyy,Melissa Rueca,Luka van Hoeven

Classifier 1: Logistic regression

Classifier 2: Feed forward

The feed forward Neural Network and its related files are found in the ./sequential folder.
The train_sequential.py file trains and saves a model on the data in the dialog_acts.dat
During training it shows its accuracy and after training a confusion matrix is shown.

infer_sequential.py is a file which infers the network with a test sentence. It uses the output types and the tokanizer so this is the same for both the training and the inferring process.

baseline_rulebased.py:The rule based system uses a dictionary with the labels acting as keys in the dictionary and the values of the dictionary contain arrays off words. The input is then matched against the words in the array, the key/label of the first match it finds is then returned as output.

Cli (Bas)

majorityclass.py:
Gives the label of the majority (inform)

Confusion matrix, accuracy
CLI tool
