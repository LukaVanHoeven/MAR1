import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import scipy
import sklearn


def build_features(
    train_text: list[str],
    test_text: list[str],
    use_char: bool,
    ngram_max: int,
    min_df: int
)-> tuple[
    scipy.sparse._csr.csr_matrix,
    scipy.sparse._csr.csr_matrix,
    sklearn.feature_extraction.text.TfidfVectorizer,
    sklearn.feature_extraction.text.TfidfVectorizer | None
]:
    """
    Makes vectorizers using tfidf and converts tekst into numeric
    representations.

    @param train_text (list[str]): List containing all training 
        utterances.
    @param test_text (list[str]): List containing all test utterances.
    @param use_char (bool): If True make a charachter level vectorizer
        else don't make one.
    @param ngram_max (int): Maximum n_gram size for the vectorizer.
    @param min_df (int): Minimum word frequency threshold.

    @return (tuple[
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        sklearn.feature_extraction.text.TfidfVectorizer,
        sklearn.feature_extraction.text.TfidfVectorizer | None]): Tuple 
        containing:
        - (scipy.sparse._csr.csr_matrix): Numeric representation of 
            `train_text`.
        - (scipy.sparse._csr.csr_matrix): Numeric representation of 
            `test_text`.
        - (sklearn.feature_extraction.text.TfidfVectorizer): Vectorizer
            for the words.
        - (sklearn.feature_extraction.text.TfidfVectorizer | None):
            Vectorizer for the characters. Can be None if `use_char` is
            True.
    """
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, ngram_max),
        lowercase=True,
        min_df=min_df
    )
    Xtr_word = word_vec.fit_transform(train_text)
    Xte_word = word_vec.transform(test_text)

    if use_char:
        char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=1
        )
        Xtr_char = char_vec.fit_transform(train_text)
        Xte_char = char_vec.transform(test_text)
        Xtr = hstack([Xtr_word, Xtr_char])
        Xte = hstack([Xte_word, Xte_char])
    else:
        char_vec = None
        Xtr, Xte = Xtr_word, Xte_word
    
    return Xtr, Xte, word_vec, char_vec

def train_logreg(
    train_csv: str,
    test_csv: str,
    out_model: str,
    use_char: bool,
    ngram_max: int,
    min_df: int
)-> str:
    """
    Train Logistic Regression on `train_csv`. Saves model bundle as
    joblib.

    @param train_csv (str): Path location of the train set.
    @param test_csv (str): Path location of the test set.
    @param out_model (str): Output path for the model.
    @param use_char (bool): If True make a charachter level vectorizer
        else don't make one.
    @param ngram_max (int): Maximum n_gram size for the vectorizer.
    @param min_df (int): Minimum word frequency threshold.

    @return (str): The file path containing the logistic regression
        joblib.
    """
    train_df, test_df = pd.read_csv(train_csv,
        keep_default_na=False), pd.read_csv(test_csv,
        keep_default_na=False)
    Xtr_text, ytr = train_df["text"].tolist(), train_df["label"].tolist()
    Xte_text = test_df["text"].tolist()

    Xtr, _, wv, cv = build_features(
        Xtr_text,
        Xte_text,
        use_char,
        ngram_max,
        min_df
    )

    clf = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced"
    )
    history = clf.fit(Xtr, ytr)

    labels_order = train_df["label"].value_counts().index.tolist()

    # Save bundle
    joblib.dump({"model": clf, "word_vec": wv, "char_vec": cv, "labels_order": labels_order}, out_model)

    return out_model, history

def logreg(data: list[str], model_path: str)-> list[str]:
    """
    Infers the logistic regression model.

    @param data (list[str]): List of all the utterances that need to be
        classified.
    @param model (str): The file path containing the logistic regression
        joblib.

    @return list[str]: List containing all the classified labels.
    """
    bundle = joblib.load(model_path)
    clf = bundle["model"]
    wv  = bundle["word_vec"]

    Xw = wv.transform(data)
    
    pred = clf.predict(Xw)
    
    return pred
