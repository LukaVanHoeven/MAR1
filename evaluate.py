import numpy as np

from sklearn.metrics import classification_report
from collections import defaultdict

from train_sequential import train_sequential
from baseline_majority_class import majority_class
from baseline_rulebased import rulebased
from ML_sequential import sequential
from ML_logreg import logreg

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