# step1_normalize.py  (parse.py)
# --------------------------------------------------------------------
# Purpose:
#   Load a simple "label + utterance" dataset from a text file,
#   normalize each utterance (lowercasing, whitespace collapsing,
#   numeric and noise token mapping), and save the result as CSV.
#
# Expected input format (dialog_acts.dat):
#   One example per line. Each line starts with a single label (no spaces),
#   followed by a space, then the raw utterance text.
#   Examples:
#       greeting hello there!
#       inform the price is 1299 dollars
#       noise <tv_noise>
#
# Output:
#   normalized.csv with two columns: label, text
# --------------------------------------------------------------------

from pathlib import Path
import re
import pandas as pd

def normalize_text(s: str) -> str:
    """
    Normalize a single utterance string.

    Steps:
      1) Lowercase the entire string (for case-insensitive modeling).
      2) Collapse any run of whitespace (spaces/tabs/newlines) to a single space.
      3) Replace standalone numbers with 3+ digits with the placeholder "<num>".
         - Rationale: Large numbers (e.g., 123, 2025) often act like IDs or prices;
           we abstract them away while keeping very small numbers (1–2 digits)
           that may carry semantics (e.g., "2 pm", "id 42").
      4) Map common noise tags to a unified placeholder "<noise>".
         - We match whole words (word boundaries) for: tv_noise, noise, sil.

    Args:
        s: Raw utterance string (may be None or empty).

    Returns:
        The normalized string.
    """
    s = (s or "").lower()
    # Collapse any sequence of whitespace characters into a single space, then trim
    s = re.sub(r"\s+", " ", s).strip()
    # Replace tokens that are *entirely* 3+ digits with <num> (word-boundary anchored)
    s = re.sub(r"\b\d{3,}\b", "<num>", s)
    # Unify noise markers such as "tv_noise", "noise", or "sil" to a single token
    s = re.sub(r"\b(?:tv_noise|noise|sil)\b", "<noise>", s)
    return s

def load_and_normalize(data_path: Path) -> pd.DataFrame:
    """
    Read the input dataset and return a DataFrame with normalized text.

    The function expects each non-empty line to start with a label,
    followed by an optional utterance string. If no utterance is present,
    an empty string is assumed.

    Args:
        data_path: Path to the input text file (e.g., dialog_acts.dat).

    Returns:
        A pandas DataFrame with columns:
          - 'label' : the first token on the line
          - 'text'  : the normalized remainder of the line (utterance)
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

if __name__ == "__main__":
    # Resolve the directory containing this script
    script_dir = Path(__file__).resolve().parent

    # Input file is expected to be in the same directory as this script
    data_path  = script_dir / "dialog_acts.dat"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {data_path}. "
            f"Place 'dialog_acts.dat' in the same directory as this script."
        )

    # Load, normalize, and write out as CSV
    df = load_and_normalize(data_path)
    out_csv = script_dir / "normalized.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # Quick, human-readable report for sanity checking
    print(f"Saved {len(df)} rows -> {out_csv}")
    print(f"Unique labels: {df['label'].nunique()}")
    print("\nTop labels:")
    print(df["label"].value_counts().head(15))
    print("\nSample:\n", df.head(10).to_string(index=False))
