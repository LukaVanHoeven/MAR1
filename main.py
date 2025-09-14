import pandas as pd
import numpy as np
from train_sequential import train_sequential
from baseline_majority_class import majority_class
from baseline_rulebased import rulebased
from ML_sequential import sequential
from ML_logreg import logreg, train_and_eval
from sklearn.metrics import classification_report
from collections import defaultdict


def main():
    # Load trainsets
    orig_train = pd.read_csv("train_orig.csv")
    dedup_train = pd.read_csv("train_dedup.csv")

    # Train
    model_sequential_orig, tokenizer_sequential_orig = train_sequential("sequential_orig", orig_train)
    model_sequential_dedup, tokenizer_sequential_dedup = train_sequential("sequential_dedup", dedup_train)

    model_logreg_orig = train_and_eval("train_orig.csv", "test_orig.csv", "logreg_orig.joblib", False, 2, 1)
    model_logreg_dedup = train_and_eval("train_dedup.csv", "test_dedup.csv", "logreg_dedup.joblib", False, 2, 1)

    # Load testsets
    orig_test = pd.read_csv("test_orig.csv")
    dedup_test = pd.read_csv("test_dedup.csv")

    # Quantitative evaluation and system comparison
    acc_rulebased_orig = accuracy_baseline(rulebased, orig_test)
    acc_rulebased_dedup = accuracy_baseline(rulebased, dedup_test)

    acc_majority_orig = accuracy_baseline(majority_class, orig_test)
    acc_majority_dedup = accuracy_baseline(majority_class, dedup_test)

    acc_sequential_orig = accuracy_ML_sequential(sequential, orig_test, model_sequential_orig, tokenizer_sequential_orig)
    acc_sequential_dedup = accuracy_ML_sequential(sequential, dedup_test, model_sequential_dedup, tokenizer_sequential_dedup)

    acc_logreg_orig = accuracy_ML_logreg(logreg, orig_test, model_logreg_orig)
    acc_logreg_dedup = accuracy_ML_logreg(logreg, dedup_test, model_logreg_dedup)

    # System comparison, chosen for accuracy as it's an easy to use metric to globally represent model performance
    print("\n---------------------\nSystem comparison:\n")
    print("Model accuracies on models trained/tested on original data")
    print(f"Rule-based baseline = {acc_rulebased_orig * 100}%")
    print(f"Majority baseline= {acc_majority_orig * 100}%")
    print(f"Sequential ML = {acc_sequential_orig * 100}%")
    print(f"Logreg ML = {acc_logreg_orig * 100}%")

    print("Model accuracies on models trained/tested on deduplicated data")
    print(f"Rule-based baseline = {acc_rulebased_dedup * 100}%")
    print(f"Majority baseline = {acc_majority_dedup * 100}%")
    print(f"Sequential ML = {acc_sequential_dedup * 100}%")
    print(f"Logreg ML = {acc_logreg_dedup * 100}%")
    print("---------------------\n")

    # Error analysis
    error_analysis(orig_test)

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

    difficult_cases(utterances_difficult_cases)

    # User input loop
    models = [
        "Rule-based baseline", 
        "Majority baseline", 
        "Sequential ML trained on original data",
        "Sequential ML trained on deduplicated data", 
        "Logreg ML trained on original data",
        "Logreg ML trained on deduplicated data"
    ]

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
                label = sequential([sentence], "sequential_orig.keras", "sequential_orig.pickle")[0]
            elif choice == "4":
                label = sequential([sentence], "sequential_dedup.keras", "sequential_dedup.pickle")[0]
            elif choice == "5":
                label = logreg([sentence], "logreg_orig.joblib")[0]
            elif choice == "6":
                label = logreg([sentence], "logreg_dedup.joblib")[0]

            print(f"The predicted label is: {label}\n\n")


def error_analysis(data):
    labels = data['label'].tolist()
    sentences = data['text'].tolist()

    results_dialogue_acts = {}
    difficult_utterances = {}

    models = [
        "Rule-based baseline", 
        "Majority baseline", 
        "Sequential ML trained on original data",
        "Sequential ML trained on deduplicated data", 
        "Logreg ML trained on original data",
        "Logreg ML trained on deduplicated data"
    ]

    predictions = np.array([
        [rulebased(sentence) for sentence in sentences],
        [majority_class(sentence) for sentence in sentences],
        sequential(sentences, "sequential_orig.keras", "sequential_orig.pickle"),
        sequential(sentences, "sequential_dedup.keras", "sequential_dedup.pickle"),
        logreg(sentences, "logreg_orig.joblib"),
        logreg(sentences, "logreg_dedup.joblib")
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


def difficult_cases(utterances_difficult_cases):

    models = [
        "Rule-based baseline", 
        "Majority baseline", 
        "Sequential ML trained on original data",
        "Sequential ML trained on deduplicated data", 
        "Logreg ML trained on original data",
        "Logreg ML trained on deduplicated data"
    ]

    utterances = list(utterances_difficult_cases.keys())
    labels = list(utterances_difficult_cases.values())

    predictions = np.array([
        [rulebased(u) for u in utterances],
        [majority_class(u) for u in utterances],
        sequential(utterances, "sequential_orig.keras", "sequential_orig.pickle"),
        sequential(utterances, "sequential_dedup.keras", "sequential_dedup.pickle"),
        logreg(utterances, "logreg_orig.joblib"),
        logreg(utterances, "logreg_dedup.joblib")
    ])

    #comparing of the same utterance across all models
    #showing the true label and each model's predicted label
    print("\n---------------------")
    print("--Difficult Cases")
    for i in range(len(models)):
        print(f"--Model: {models[i]}")
        pred_system = predictions[i] 
        for j in range(len(utterances)):
            print(f"Utterance: {utterances[j]}, Label: {labels[j]} Prediction: {pred_system[j]}")
        

def accuracy_baseline(func, testset):
    correct = 0
    for _, row in testset.iterrows():
        if func(row['text']) == row['label']:
            correct += 1

    return round(correct / testset.shape[0], 4)


def accuracy_ML_sequential(func, testset, model, tokenizer):
    predictions = func(testset["text"].tolist(), model, tokenizer)
    
    correct = 0

    for pred, label in zip(predictions, testset["label"].tolist()):
        if pred == label:
            correct += 1

    return round(correct / len(predictions), 4)


def accuracy_ML_logreg(func, testset, model):
    predictions = func(testset["text"].tolist(), model)

    correct = 0

    for pred, label in zip(predictions, testset["label"].tolist()):
        if pred == label:
            correct += 1

    return round(correct / len(predictions), 4)



if __name__ == "__main__":
    main()
    