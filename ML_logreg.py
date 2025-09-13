# ML_logreg.py
# Logistic Regression (OvR) for dialog-act classification on TF-IDF features.

import argparse
from pathlib import Path
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

# -utilities-
def read_split(train_csv: str, test_csv: str):
    """
    Load train/test CSVs (columns: label, text). Basic cleanup only.
    """
    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)

    train = train.dropna(subset=["label", "text"])
    test  = test.dropna(subset=["label", "text"])

    train["label"] = train["label"].astype(str).str.strip()
    test["label"]  = test["label"].astype(str).str.strip()
    train["text"]  = train["text"].astype(str).str.strip()
    test["text"]   = test["text"].astype(str).str.strip()
    return train, test

def build_features(train_text, test_text, use_char: bool, ngram_max: int, min_df: int):
    """
    Create TF-IDF features:
      - word 1..ngram_max
      - optional char 3..5
    Returns: Xtr, Xte, word_vec, char_vec (char_vec may be None).
    """
    word_vec = TfidfVectorizer(analyzer="word",
                               ngram_range=(1, ngram_max),
                               lowercase=True,
                               min_df=min_df)
    Xtr_word = word_vec.fit_transform(train_text)
    Xte_word = word_vec.transform(test_text)

    if use_char:
        char_vec = TfidfVectorizer(analyzer="char",
                                   ngram_range=(3, 5),
                                   lowercase=True,
                                   min_df=1)
        Xtr_char = char_vec.fit_transform(train_text)
        Xte_char = char_vec.transform(test_text)
        Xtr = hstack([Xtr_word, Xtr_char])
        Xte = hstack([Xte_word, Xte_char])
    else:
        char_vec = None
        Xtr, Xte = Xtr_word, Xte_word

    return Xtr, Xte, word_vec, char_vec

def train_and_eval(train_csv: str, test_csv: str, out_model: str,
                   use_char: bool, ngram_max: int, min_df: int):
    """
    Train Logistic Regression on train_csv and evaluate on test_csv.
    Saves model bundle as joblib.
    """
    train_df, test_df = read_split(train_csv, test_csv)
    Xtr_text, ytr = train_df["text"].tolist(), train_df["label"].tolist()
    Xte_text, yte = test_df["text"].tolist(),  test_df["label"].tolist()

    Xtr, Xte, wv, cv = build_features(Xtr_text, Xte_text, use_char, ngram_max, min_df)

    clf = LogisticRegression(solver="liblinear",
                             max_iter=1000,
                             class_weight="balanced")
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro", zero_division=0)

    print("\n=== Logistic Regression (simple) ===")
    print(f"Train file : {train_csv}")
    print(f"Test  file : {test_csv}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Macro-F1   : {f1m:.4f}\n")

    labels_order = train_df["label"].value_counts().index.tolist()
    print(classification_report(yte, ypred, labels=labels_order, zero_division=0))

    # Save bundle next to the script (relative path)
    out_path = Path(out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "word_vec": wv, "char_vec": cv, "labels_order": labels_order}, out_path)
    print(f"\nSaved model -> {out_path}")

    # Also save confusion matrix CSV (handy for report)
    cm = confusion_matrix(yte, ypred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in labels_order],
                            columns=[f"pred:{l}" for l in labels_order])
    cm_csv = out_path.with_name(out_path.stem + "_cm.csv")
    cm_df.to_csv(cm_csv, index=True)
    print(f"Confusion matrix -> {cm_csv}")

def infer_loop(model_path: str, topk: int):
    """
    Load saved model (.joblib) and run an interactive prediction loop.
    """
    bundle = joblib.load(model_path)
    clf = bundle["model"]
    wv  = bundle["word_vec"]
    cv  = bundle["char_vec"]

    print("Interactive mode. Type an utterance (or /quit to exit).")
    while True:
        try:
            s = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if s.lower() in {"/quit", "/exit"}:
            print("bye.")
            break

        Xw = wv.transform([s])
        if cv is not None:
            from scipy.sparse import hstack
            Xc = cv.transform([s])
            X  = hstack([Xw, Xc])
        else:
            X = Xw

        pred = clf.predict(X)[0]
        print("pred:", pred)

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            classes = clf.classes_
            pairs = sorted(zip(classes, proba), key=lambda t: t[1], reverse=True)[:topk]
            print("top-{}: {}".format(topk, ", ".join(f"{c}:{p:.3f}" for c, p in pairs)))

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Logistic Regression for dialog acts (single-file, repo-relative).")
    ap.add_argument("--mode", choices=["train", "infer"], default="train",
                    help="train (default) or infer")
    ap.add_argument("--train_file", default="train_orig.csv",
                    help="train CSV (default: train_orig.csv)")
    ap.add_argument("--test_file", default="test_orig.csv",
                    help="test CSV (default: test_orig.csv)")
    ap.add_argument("--model_out", default="logreg_basic.joblib",
                    help="output model path for --mode train (default: logreg_basic.joblib)")
    ap.add_argument("--model_in", default="logreg_basic.joblib",
                    help="input model path for --mode infer (default: logreg_basic.joblib)")
    ap.add_argument("--use_char", action="store_true",
                    help="add char 3-5 n-grams features")
    ap.add_argument("--ngram_max", type=int, default=2,
                    help="max word n-gram size (default 2)")
    ap.add_argument("--min_df", type=int, default=1,
                    help="min_df for TF-IDF (default 1)")
    ap.add_argument("--topk", type=int, default=3,
                    help="top-k classes to display in infer mode (if proba is available)")
    args = ap.parse_args()

    if args.mode == "train":
        train_and_eval(args.train_file, args.test_file,
                       args.model_out, args.use_char, args.ngram_max, args.min_df)
    else:
        infer_loop(args.model_in, args.topk)

if __name__ == "__main__":
    main()
