import pandas as pd
from train_sequential import train_sequential
from baseline_majority_class import majority_class
from baseline_rulebased import rulebased
from ML_sequential import sequential
from ML_logreg import logreg, train_and_eval


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

    # System comparison
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

    # Error analysis
    
    
    # Difficult cases
    

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
    