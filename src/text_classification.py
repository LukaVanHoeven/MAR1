from .evaluate import accuracy_baseline, accuracy_ML, error_analysis, difficult_cases
from .logistic_regression import train_logreg, logreg
from .majority_class import majority_class
from .parse import parse
from .rulebased import rulebased
from .sequential import train_sequential, sequential
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def assignment1a()-> None:
    """
    Executes the code needed for assignment 1a of the INFOMAIR course of
    Utrecht Universities AI master. This assignment requires you to 
    write baseline or train Machine Learning classification models. The 
    classification concerns classifying user utterances into different
    dialogue acts for restaurant recommendation. This code also 
    evaluates these models to allow you to make a decision on which
    model you prefer.

    The code does the following:
        - Parses the dataset
        - Loads the dataset
        - Trains the Machine learning models
        - Performs quantative evaluation
        - Performs system comparison
        - Performs error analysis
        - Tests all models on difficult cases
        - Allows users to test all models with their own input
    """
    # Parse the dataset
    parse()

    # Load trainsets
    data_folder = Path(__file__).resolve().parent.parent / "data"
    orig_train = pd.read_csv(
        data_folder / "train_orig.csv",
        keep_default_na=False
    )
    dedup_train = pd.read_csv(
        data_folder / "train_dedup.csv",
        keep_default_na=False
    )

    # Train
    model_folder = Path(__file__).resolve().parent.parent / "models"
    model_sequential_orig, tokenizer_sequential_orig, history_sequential_orig = train_sequential(
        model_folder / "sequential_orig",
        orig_train
    )
    
    f1_score_sequential_orig = history_sequential_orig.history['val_accuracy'][-1]
    plot_loss(history_sequential_orig)
    
    model_sequential_dedup, tokenizer_sequential_dedup, history_sequential_dedup = train_sequential(
        model_folder / "sequential_dedup",
        dedup_train
    )
    
    f1_score_sequential_dedup = history_sequential_dedup.history['val_accuracy'][-1]
    plot_loss(history_sequential_dedup)
    model_logreg_orig = train_logreg(
        data_folder / "train_orig.csv",
        data_folder / "test_orig.csv",
        model_folder / "logreg_orig.joblib",
        False,
        1,
        1
    )
    model_logreg_dedup = train_logreg(
        data_folder / "train_dedup.csv",
        data_folder / "test_dedup.csv",
        model_folder / "logreg_dedup.joblib",
        False,
        1,
        1
    )

    # Load testsets
    orig_test = pd.read_csv(
        data_folder / "test_orig.csv",
        keep_default_na=False
    )
    dedup_test = pd.read_csv(
        data_folder / "test_dedup.csv",
        keep_default_na=False
    )

    # Quantitative evaluation and system comparison
    acc_rulebased_orig = accuracy_baseline(rulebased, orig_test)
    acc_rulebased_dedup = accuracy_baseline(rulebased, dedup_test)

    acc_majority_orig = accuracy_baseline(majority_class, orig_test)
    acc_majority_dedup = accuracy_baseline(majority_class, dedup_test)

    acc_sequential_orig, cm_sequential_orig = accuracy_ML(sequential, orig_test, model_sequential_orig, tokenizer_sequential_orig, True)
    acc_sequential_dedup, cm_sequential_dedup = accuracy_ML(sequential, dedup_test, model_sequential_dedup, tokenizer_sequential_dedup, True)
    acc_logreg_orig, cm_logreg_orig = accuracy_ML(logreg, orig_test, model_logreg_orig, None, False)
    acc_logreg_dedup, cm_logreg_dedup = accuracy_ML(logreg, dedup_test, model_logreg_dedup, None, False)

    # System comparison, chosen for accuracy as it's an easy to use metric to globally represent model performance
    print("\n---------------------\nSystem comparison:\n")
    print("Model accuracies on models trained/tested on original data")
    print(f"Rule-based baseline = {acc_rulebased_orig * 100}%")
    print(f"Majority baseline= {acc_majority_orig * 100}%")
    print(f"Sequential ML = {acc_sequential_orig * 100}%")
    print(f"Sequential ML F1-score = {f1_score_sequential_orig * 100}%")
    print(f"Logreg ML = {acc_logreg_orig * 100}%")

    print("Model accuracies on models trained/tested on deduplicated data")
    print(f"Rule-based baseline = {acc_rulebased_dedup * 100}%")
    print(f"Majority baseline = {acc_majority_dedup * 100}%")
    print(f"Sequential ML = {acc_sequential_dedup * 100}%")
    print(f"Sequential ML F1-score = {f1_score_sequential_dedup * 100}%")
    print(f"Logreg ML = {acc_logreg_dedup * 100}%")
    print("---------------------\n")

    models = [
        "Rule-based baseline", 
        "Majority baseline", 
        "Sequential ML trained on original data",
        "Sequential ML trained on deduplicated data", 
        "Logreg ML trained on original data",
        "Logreg ML trained on deduplicated data"
    ]

    # Error analysis
    error_analysis(orig_test, models)

    # Difficult cases
    #list of examples of difficult cases
    #two types: mispelling and negation
    utterances_difficult_cases = {
        "cna you find an italian restaurant" : "request", 
        "coul u send me the adress?" : "request",
        "i need mre informations" : "reqmore",
        "i don't want italian restaurant" : "deny",
        "i don't want other information" : "deny",
        "i don't need other suggestions" : "deny"
    }

    difficult_cases(utterances_difficult_cases, models)

    # User input loop
    while True:
        print("Type the number corresponding to your preferred model or type 'exit' to close the programme:")
        for i, model in enumerate(models):
            print(f"({i + 1}) {model}")

        choice = input("")

        if choice == "exit":
            print("Closing the programme!")
            return
        try:
            if int(choice) not in range(1, len(models) + 1):
                print("Invalid choice, try again\n\n")
                continue
        except ValueError:
            print("Invalid choice, try again\n\n")
            continue

        while True:
            sentence = input("Type your sentence here (or press 's' to go back to the model selection options or press 'exit' to close the programme):\n").lower()

            if sentence == "exit":
                print("Closing the programme!")
                return
            elif sentence == "s":
                break
            elif choice == "1":
                label = rulebased(sentence)
            elif choice == "2":
                label = majority_class(sentence)
            elif choice == "3":
                label = sequential([sentence], model_folder / "sequential_orig.keras", model_folder / "sequential_orig.pickle")[0]
            elif choice == "4":
                label = sequential([sentence], model_folder / "sequential_dedup.keras", model_folder / "sequential_dedup.pickle")[0]
            elif choice == "5":
                label = logreg([sentence], model_folder / "logreg_orig.joblib")[0]
            elif choice == "6":
                label = logreg([sentence], model_folder / "logreg_dedup.joblib")[0]

            print(f"The predicted label is: {label}\n\n")

def show_confusion_matrix(cm):
    """
    Displays the confusion matrix in a readable format.

    @param cm: The confusion matrix to display.
    """
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
    labels = [k for k, v in sorted(output_types.items(), key=lambda x: x[1])]
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel('True')
    plt.show()
    

def plot_loss(history):
    """
    Plots the training and validation loss from a Keras model history object.
    
    Parameters:
        history: The History object returned by model.fit(...)
    """
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Training Loss')
    
    if 'val_loss' in history.history:  # only plot if validation loss exists
        plt.plot(history.history['val_loss'], label='Validation Loss')
    
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()