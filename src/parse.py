from pathlib import Path
from sklearn.utils import check_random_state
import os
import pandas as pd
import re


def normalize_text(utterance: str)-> str:
    """
    Normalize a single utterance. According to the following steps:
      - Lowercase the entire string (for case-insensitive modeling).
      - Collapse any run of whitespace (spaces/tabs/newlines) to a 
        single space.
      - Replace standalone numbers with 3+ digits with the placeholder
        "<num>".
         - Rationale: Large numbers (e.g., 123, 2025) often act like IDs
            or prices; we abstract them away while keeping very small
            numbers (1 or 2 digits) that may carry semantics (e.g., 
            "2 pm", "id 42").

    @param utterance (str): Raw utterance string (may be None or empty).

    @return (str): The normalized string.
    """
    utterance = (utterance or "").lower()
    # Collapse any sequence of whitespace characters into a single space, then trim
    utterance = re.sub(r"\s+", " ", utterance).strip()
    # Replace tokens that are *entirely* 3+ digits with <num> (word-boundary anchored)
    utterance = re.sub(r"\b\d{3,}\b", "<num>", utterance)
    return utterance

def load_and_normalize(data_path: Path)-> pd.DataFrame:
    """
    Read the input dataset and return a DataFrame with normalized text.

    The function expects each line to start with a label, followed by an
    utterance string. If no utterance is present, an empty string is 
    assumed.

    @param data_path (Path): Path to the input text file.

    @return (pd.Dataframe): Dataframe containing all user utterances
        (now normalized) and their corresponding labels.
    """
    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                # Skip blank lines to avoid emitting empty rows
                continue
            # Split into at most two parts: label and the rest (utterance)
            parts = line.split(maxsplit=1)
            label = parts[0]
            text  = parts[1] if len(parts) == 2 else ""
            rows.append((label, normalize_text(text)))
    return pd.DataFrame(rows, columns=["label", "text"])

def majority_label_safe(label_series: pd.Series)-> str:
    """
    Determines the label that appears most frequent in a pandas series
    object. In case of a tie the first label is chosen.

    @param label_series (pd.Series): pandas series object containing all
        the labels.

    @return (str): The found label.
    """
    vc = label_series.value_counts()
    top = vc.max()
    candidates = vc[vc == top].index.tolist()
    return candidates[0]

def split_df(
    df: pd.DataFrame,
    label_col: str="label",
    test_size: float=0.15,
    seed: int=42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataframe into a train set and test set.

    @param df (pd.Dataframe): Dataframe containing all user utterances
        and their corresponding labels.
    @param label_col (str): Name of the column containing the labels.
    @param test_size (float): Partition of the data set that needs to go
        into the test set.
    @param seed (int): Seed for randomisation.

    @return tuple[pd.DataFrame, pd.DataFrame]: Tuple containing:
        - (pd.Dataframe): Dataframe containing the train set.
        - (pd.Dataframe): Dataframe containing the test set.
    """
    rng = check_random_state(seed)
    train_parts, test_parts = [], []
    for _, grp in df.groupby(label_col, sort=False):
        n = len(grp)
        if n == 1:
            train_parts.append(grp)
        elif n == 2:
            idx = rng.choice(grp.index, size=1, replace=False)
            test_parts.append(grp.loc[idx])
            train_parts.append(grp.drop(idx))
        else:
            n_test = max(1, min(n - 1, int(round(n * test_size))))
            test_idx = rng.choice(grp.index, size=n_test, replace=False)
            test_parts.append(grp.loc[test_idx])
            train_parts.append(grp.drop(test_idx))

    train_df = pd.concat(
        train_parts
    ).sample(
        frac=1.0,
        random_state=seed
    ).reset_index(
        drop=True
    )

    test_df  = pd.concat(
        test_parts
    ).sample(
        frac=1.0,
        random_state=seed
    ).reset_index(
        drop=True
    )
    return train_df, test_df

def parse(
    file_name: str="dialog_acts.dat",
    test_size: float=0.15
)-> None:
    """
    Parse the data file so you'll end up with a train and test set for
    a deduplicated dataset and the not deduplicated dataset.

    Steps:
        - Load the data into a pandas dataframe.
        - Normalize the data.
        - Deduplicate the data into a seperate dataframe.
        - Split both dataframes into a train and test set according to
        `test_size`.

    @param file_name (str): Name of the file containing the data.
    @param test_size (float): Partition of the data set that needs to go
        into the test set.
    """
    data_folder = Path(__file__).resolve().parent.parent / "data"
    data_path = data_folder / f"{file_name}"
    assert os.path.exists(data_path), f"{data_path} does not exist"

    orig_df = load_and_normalize(data_path)
    
    dedup_df = (
        orig_df.groupby(
            "text",
            dropna=False
        )["label"].apply(
            majority_label_safe
        ).reset_index().rename(
            columns={"label": "label"}
        )
    )            

    orig_train, orig_test = split_df(
        orig_df,
        "label",
        test_size,
        42
    )

    dedup_train, dedup_test = split_df(
        dedup_df,
        "label",
        test_size,
        42
    )

    orig_train.to_csv(
        data_folder / "train_orig.csv",
        index=False
    )
    orig_test.to_csv(
        data_folder / "test_orig.csv",
        index=False
    )
    dedup_train.to_csv(
        data_folder / "train_dedup.csv",
        index=False
    )
    dedup_test.to_csv(
        data_folder / "test_dedup.csv",
        index=False
    )
