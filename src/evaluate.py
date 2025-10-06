from .majority_class import majority_class
from .rulebased import rulebased
from .sequential import sequential
from .logistic_regression import logreg
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def error_analysis(data: pd.DataFrame, models: list[str]) -> None:
    """
    performs the error analysis on the given `data`, for the given 
    `models`. The error analysis consists of:
        - Finding the hardest dialogue acts to classify for each model.
        - Finding the hardest utterances to classify for each model.
        - Finding the hardest dialogue acts to classify for all models.
        - Finding the hardest utterances to classify for all models.
        - Analyzing the relationship between utterance length and errors.
    
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

    # NEW: Utterance length analysis
    length_analysis_results = analyze_utterance_length(
        sentences, labels, predictions, models
    )

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

    # NEW: Print utterance length analysis
    print("\n---------------------")
    print("--UTTERANCE LENGTH ANALYSIS")
    print_length_analysis(length_analysis_results, models)
    
    # NEW: Optional - create visualizations
    plot_length_analysis(length_analysis_results, models)


def analyze_utterance_length(
    sentences: list[str],
    labels: list[str],
    predictions: np.ndarray,
    models: list[str]
) -> dict:
    """
    Analyzes the relationship between utterance length and classification errors.
    
    @param sentences (list[str]): List of all utterances
    @param labels (list[str]): List of true labels
    @param predictions (np.ndarray): Array of predictions from all models
    @param models (list[str]): List of model names
    @return dict: Dictionary containing length analysis results
    """
    # Calculate utterance lengths (in words)
    lengths = [len(s.split()) for s in sentences]
    
    results = {
        'overall_stats': {},
        'per_model': {},
        'length_bins': {}
    }
    
    # Overall statistics
    results['overall_stats'] = {
        'avg_length_all': np.mean(lengths),
        'std_length_all': np.std(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths)
    }
    
    # Per-model analysis
    for i, model in enumerate(models):
        model_predictions = predictions[i]
        
        # Separate correct and incorrect predictions
        correct_indices = [j for j in range(len(labels)) if labels[j] == model_predictions[j]]
        incorrect_indices = [j for j in range(len(labels)) if labels[j] != model_predictions[j]]
        
        correct_lengths = [lengths[j] for j in correct_indices]
        incorrect_lengths = [lengths[j] for j in incorrect_indices]
        
        results['per_model'][model] = {
            'avg_length_correct': np.mean(correct_lengths) if correct_lengths else 0,
            'avg_length_incorrect': np.mean(incorrect_lengths) if incorrect_lengths else 0,
            'std_length_correct': np.std(correct_lengths) if correct_lengths else 0,
            'std_length_incorrect': np.std(incorrect_lengths) if incorrect_lengths else 0,
            'num_correct': len(correct_indices),
            'num_incorrect': len(incorrect_indices),
            'correct_lengths': correct_lengths,
            'incorrect_lengths': incorrect_lengths
        }
    
    # Length bin analysis (group by length ranges)
    length_bins = {
        '1-2 words': (1, 2),
        '3-4 words': (3, 4),
        '5-6 words': (5, 6),
        '7+ words': (7, float('inf'))
    }
    
    for bin_name, (min_len, max_len) in length_bins.items():
        bin_indices = [
            i for i in range(len(lengths)) 
            if min_len <= lengths[i] <= max_len
        ]
        
        if not bin_indices:
            continue
            
        results['length_bins'][bin_name] = {
            'count': len(bin_indices),
            'percentage': len(bin_indices) / len(lengths) * 100
        }
        
        # Error rate per model for this length bin
        for i, model in enumerate(models):
            model_predictions = predictions[i]
            errors_in_bin = sum(
                1 for j in bin_indices 
                if labels[j] != model_predictions[j]
            )
            error_rate = errors_in_bin / len(bin_indices) if bin_indices else 0
            
            if 'models' not in results['length_bins'][bin_name]:
                results['length_bins'][bin_name]['models'] = {}
            
            results['length_bins'][bin_name]['models'][model] = {
                'errors': errors_in_bin,
                'error_rate': error_rate
            }
    
    return results


def print_length_analysis(results: dict, models: list[str]) -> None:
    """
    Prints the utterance length analysis results in a readable format.
    
    @param results (dict): Results from analyze_utterance_length
    @param models (list[str]): List of model names
    """
    print("\n=== OVERALL LENGTH STATISTICS ===")
    stats = results['overall_stats']
    print(f"Average utterance length: {stats['avg_length_all']:.2f} words")
    print(f"Standard deviation: {stats['std_length_all']:.2f}")
    print(f"Range: {stats['min_length']} - {stats['max_length']} words")
    
    print("\n=== LENGTH ANALYSIS PER MODEL ===")
    for model in models:
        model_stats = results['per_model'][model]
        print(f"\n--{model}:")
        print(f"  Correct predictions: {model_stats['num_correct']}")
        print(f"    Avg length: {model_stats['avg_length_correct']:.2f} words (±{model_stats['std_length_correct']:.2f})")
        print(f"  Incorrect predictions: {model_stats['num_incorrect']}")
        print(f"    Avg length: {model_stats['avg_length_incorrect']:.2f} words (±{model_stats['std_length_incorrect']:.2f})")
        
        if model_stats['num_incorrect'] > 0:
            diff = model_stats['avg_length_incorrect'] - model_stats['avg_length_correct']
            print(f"  Difference: {diff:+.2f} words (incorrect vs correct)")
    
    print("\n=== ERROR RATE BY LENGTH BINS ===")
    for bin_name in results['length_bins']:
        bin_data = results['length_bins'][bin_name]
        print(f"\n--{bin_name}:")
        print(f"  Count: {bin_data['count']} utterances ({bin_data['percentage']:.1f}%)")
        print("  Error rates:")
        for model in models:
            if model in bin_data['models']:
                model_data = bin_data['models'][model]
                print(f"    {model}: {model_data['error_rate']*100:.2f}% ({model_data['errors']}/{bin_data['count']})")


def plot_length_analysis(results: dict, models: list[str]) -> None:
    """
    Creates visualizations for the utterance length analysis.
    
    @param results (dict): Results from analyze_utterance_length
    @param models (list[str]): List of model names
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Average length comparison (correct vs incorrect)
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    
    correct_avgs = [results['per_model'][m]['avg_length_correct'] for m in models]
    incorrect_avgs = [results['per_model'][m]['avg_length_incorrect'] for m in models]
    
    ax1.bar(x - width/2, correct_avgs, width, label='Correct', alpha=0.8)
    ax1.bar(x + width/2, incorrect_avgs, width, label='Incorrect', alpha=0.8)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Average Length (words)')
    ax1.set_title('Average Utterance Length: Correct vs Incorrect Predictions')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Error rate by length bins
    ax2 = axes[0, 1]
    bins = list(results['length_bins'].keys())
    
    for model in models:
        error_rates = [
            results['length_bins'][bin_name]['models'][model]['error_rate'] * 100
            for bin_name in bins if model in results['length_bins'][bin_name]['models']
        ]
        ax2.plot(bins[:len(error_rates)], error_rates, marker='o', label=model)
    
    ax2.set_xlabel('Utterance Length Bin')
    ax2.set_ylabel('Error Rate (%)')
    ax2.set_title('Error Rate by Utterance Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of lengths for correct vs incorrect
    ax3 = axes[1, 0]
    
    # Pick one model for detailed distribution (e.g., best performing)
    model_to_plot = models[2] if len(models) > 2 else models[0]  # Adjust as needed
    correct_lens = results['per_model'][model_to_plot]['correct_lengths']
    incorrect_lens = results['per_model'][model_to_plot]['incorrect_lengths']
    
    ax3.hist([correct_lens, incorrect_lens], bins=range(1, 15), 
             label=['Correct', 'Incorrect'], alpha=0.7)
    ax3.set_xlabel('Utterance Length (words)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Length Distribution: {model_to_plot}')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Distribution of utterances across length bins
    ax4 = axes[1, 1]
    bin_counts = [results['length_bins'][b]['count'] for b in bins]
    bin_percentages = [results['length_bins'][b]['percentage'] for b in bins]
    
    ax4.bar(bins, bin_counts, alpha=0.8)
    ax4.set_xlabel('Utterance Length Bin')
    ax4.set_ylabel('Number of Utterances')
    ax4.set_title('Distribution of Utterances by Length')
    
    # Add percentage labels on bars
    for i, (count, pct) in enumerate(zip(bin_counts, bin_percentages)):
        ax4.text(i, count, f'{pct:.1f}%', ha='center', va='bottom')
    
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('utterance_length_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'utterance_length_analysis.png'")
    plt.show()


# Keep the existing functions unchanged
def difficult_cases(
    utterances_difficult_cases: dict[str:str],
    models: list[str]
) -> None:
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
) -> float:
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
    f1_score_model = f1_score(testset["label"].tolist(), [func(row['text']) for _, row in testset.iterrows()], average='weighted', zero_division=0)
    print("F1score for baseline model", f1_score_model)
    return round(correct / testset.shape[0], 4)


def accuracy_ML(
    func: Callable[[list[str], str, str], list[str]],
    testset: pd.DataFrame,
    model: str,
    tokenizer: str = "sequential_dedup.pickle",
    sequential: bool = True
) -> tuple[float, any]:
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
    cm = confusion_matrix(testset["label"].tolist(), predictions)
    f1_score_model = f1_score(testset["label"].tolist(), predictions, average='weighted', zero_division=0)
    print("F1score for model", model, f1_score_model)
    return round(correct / len(predictions), 4), cm