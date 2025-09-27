from .majority_class import majority_class
from .rulebased import rulebased
from .sequential import sequential
from .logistic_regression import logreg
from collections import defaultdict
from sklearn.metrics import classification_report
from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd


def error_analysis(data: pd.DataFrame, models: list[str])-> None:
    """
    performs the error analysis on the given `data`, for the given 
    `models`. The error analysis consists of:
        - Finding the hardest dialogue acts to classify for each model.
        - Finding the hardest utterances to classify for each model.
        - Finding the hardest dialogue acts to classify for all models.
        - Finding the hardest utterances to classify for all models.
    
    @param data (pd.Dataframe): Dataframe containing all user utterances
        and their corresponding labels.
    @param models (list[str]): List of all model names that need to be
        evaluated.
    """
    labels = data['label'].tolist()
    sentences = data['text'].tolist()

    results_dialogue_acts = {}
    difficult_utterances = {}

    model_folder = Path(__file__).resolve().parent.parent / "models"

    predictions = np.array([
        [rulebased(sentence) for sentence in sentences],
        [majority_class(sentence) for sentence in sentences],
        sequential(
            sentences,
            model_folder / "sequential_orig.keras",
            model_folder / "sequential_orig.pickle"
        ),
        sequential(
            sentences,
            model_folder / "sequential_dedup.keras",
            model_folder / "sequential_dedup.pickle"
        ),
        logreg(sentences, model_folder / "logreg_orig.joblib"),
        logreg(sentences, model_folder / "logreg_dedup.joblib")
    ])
    
    for model, prediction in zip(models, predictions):
        report = classification_report(labels, prediction, output_dict=True, zero_division=0)
        
        # Difficult dialogue acts per model
        # Chosen for f1 since it takes both recall and precision into account
        sorted_report = sorted(
            [
                (
                    dialogue_act, report[dialogue_act]['f1-score']
                ) for dialogue_act in report if dialogue_act not in (
                    'accuracy',
                    'macro avg',
                    'weighted avg'
                )
            ],
            key=lambda x: x[1]
        )

        results_dialogue_acts[model] = sorted_report

        # Difficult utterances per model
        wrong_utterances = [
            (
                sentences[i],
                labels[i],
                prediction[i]
            ) for i in range(len(labels)) if labels[i] != prediction[i]
        ]
        difficult_utterances[model] = wrong_utterances
    
    # Difficult dialogue_acts for all models combined
    dialogue_act_scores = []
    for model in models:
        dialogue_act_scores.extend(results_dialogue_acts[model])

    avg_scores = defaultdict(list)
    for dialogue_act, score in dialogue_act_scores:
        avg_scores[dialogue_act].append(score)

    avg_scores = {
        dialogue_act: sum(scores) / len(scores) for dialogue_act, scores in avg_scores.items()
    }
    avg_scores_sorted = sorted(avg_scores.items(), key=lambda x: x[1])

    # Difficult utterances for all models
    hard_utterances = []
    for i, label in enumerate(labels):
        all_sentence_preds = predictions[:, i]
        if all(pred != label for pred in all_sentence_preds):
            hard_utterances.append((sentences[i], label, all_sentence_preds))

    # Printing
    # Hardest dialogue acts per model
    print("\n---------------------")
    for model in models:
        print(f"--Hardest dialogue act for {model}:")
        for dialogue_act, score in results_dialogue_acts[model]:
            print(f"{dialogue_act} -- F1 = {score}")

    # Hardest utterances per model
    print("\n---------------------")
    for model in models:
        print(f"--Hardest utterances for {model}")
        # Only show the 3 most difficult
        for utterance, label, prediction in difficult_utterances[model][:3]:
            print(f"Utterance: {utterance}\nLabel: {label}, Prediction: {prediction}\n")

    # Hardest dialogue acts for all models
    print("\n---------------------")
    print("--Hardest dialogue acts for all models")
    for dialogue_act, score in avg_scores_sorted:
        print(f"{dialogue_act} -- average F1 = {score}")
    
    # Utterances that went wrong for all models
    print("\n---------------------")
    print("--Utterances that went wrong for all models")
    for utterance, label, prediction in hard_utterances:
        print(f"Utterance: {utterance}\nLabel: {label}, Prediction: {prediction}\n")


def difficult_cases(
    utterances_difficult_cases: dict[str:str],
    models: list[str]
)-> None:
    """
    Evaluate all the given `models` on the given difficult cases. This 
    function prints per model the user utterance, the expected label and
    the label that was predicted by the model.
    
    @param utterances_difficult_cases (dict[str:str]): Dictionary 
        containing a difficult user utterance as it's keys and the 
        corresponding expected label as values.
    @param models (list[str]): List of all model names that need to be
        evaluated.
    """
    utterances = list(utterances_difficult_cases.keys())
    labels = list(utterances_difficult_cases.values())

    model_folder = Path(__file__).resolve().parent.parent / "models"

    predictions = np.array([
        [rulebased(u) for u in utterances],
        [majority_class(u) for u in utterances],
        sequential(
            utterances,
            model_folder / "sequential_orig.keras",
            model_folder / "sequential_orig.pickle"
        ),
        sequential(utterances,
            model_folder / "sequential_dedup.keras",
            model_folder / "sequential_dedup.pickle"
        ),
        logreg(utterances, model_folder / "logreg_orig.joblib"),
        logreg(utterances, model_folder / "logreg_dedup.joblib")
    ])

    #comparing of the same utterance across all models
    #showing the true label and each model's predicted label
    print("\n---------------------")
    print("--Difficult Cases")
    for i in range(len(models)):
        print(f"--Model: {models[i]}")
        pred_system = predictions[i] 
        for j in range(len(utterances)):
            print(
                f"Utterance: {utterances[j]}, Label: {labels[j]} " \
                f"Prediction: {pred_system[j]}"
            )
        

def accuracy_baseline(
    func: Callable[[str], str],
    testset: pd.DataFrame
)-> float:
    """
    Calculates the accuracy of a baseline model.

    @param func (Callable[[str], str]): Callable function that accepts
        a user utterance and returns a classification label.
    @param testset (pd.DataFrame): Dataframe containing the testset
        needed to evaluate the baseline model. Contains a text column
        and a label column.
    """
    correct = 0
    for _, row in testset.iterrows():
        if func(row['text']) == row['label']:
            correct += 1

    return round(correct / testset.shape[0], 4)


def accuracy_ML(
    func: Callable[[list[str], str, str], list[str]],
    testset: pd.DataFrame,
    model: str,
    tokenizer: str="sequential_dedup.pickle",
    sequential: bool=True
)-> float:
    """
    Calculates the accuracy of a either a sequential or a logistic 
    regression machine learning model.

    @param func (Callable[[list[str], str, str]): Callable function that
        accepts a list of user utterance and the location of the trained
        model and finally returns a list of classification labels.
    @param testset (pd.DataFrame): Dataframe containing the testset
        needed to evaluate the baseline model. Contains a text column
        and a label column.
    @param model (str): The path location of the model that's to be 
        used.
    @param tokenizer (str): The path location of the tokenizer for the
        sequential model.
    @param sequential (bool): If True calculate the accuracy of a
        sequential model. If False do so for a linear regression model.
    """
    if sequential:
        predictions = func(testset["text"].tolist(), model, tokenizer)
    else: 
        predictions = func(testset["text"].tolist(), model)
    
    correct = 0

    for pred, label in zip(predictions, testset["label"].tolist()):
        if pred == label:
            correct += 1

    return round(correct / len(predictions), 4)
