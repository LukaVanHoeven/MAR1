# step3_split.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from sklearn.utils import check_random_state

HERE = Path(__file__).resolve().parent
SEED = 42
TEST_SIZE = 0.15

@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame

def stratified_split_safe(df: pd.DataFrame, label_col="label",
                          test_size=0.15, seed=42) -> SplitResult:
    """
    Стратифицированный и безопасный сплит:
    - n=1: всё в train
    - n=2: 1 в train, 1 в test
    - n>=3: округляем тестовую долю, но оставляем минимум 1 в train и 1 в test
    """
    rng = check_random_state(seed)
    train_parts, test_parts = [], []
    for lab, grp in df.groupby(label_col, sort=False):
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

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df  = pd.concat(test_parts ).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return SplitResult(train_df, test_df)

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"label": "string", "text": "string"})
    # deffending from the null
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].str.strip()
    df["text"]  = df["text"].str.strip()
    df = df[(df["label"] != "") & (df["text"] != "")]
    return df

def save_split(res: SplitResult, prefix: str):
    (HERE / f"train_{prefix}.csv").write_text("", encoding="utf-8")  # ensure writable on some OSes
    res.train.to_csv(HERE / f"train_{prefix}.csv", index=False)
    res.test.to_csv (HERE / f"test_{prefix}.csv",  index=False)

def print_dist(df: pd.DataFrame, title: str):
    print(f"\n{title} (n={len(df)}):")
    print(df["label"].value_counts().head(15))

def main():
    # --- ORIG (normalized.csv) ---
    orig_path = HERE / "normalized.csv"
    if not orig_path.exists():
        raise FileNotFoundError("normalized.csv не найден. Запусти шаг 1 (нормализация).")

    orig = load_csv(orig_path)
    print(f"ORIG loaded: {len(orig)} rows, {orig['label'].nunique()} labels")
    split_orig = stratified_split_safe(orig, test_size=TEST_SIZE, seed=SEED)
    save_split(split_orig, "orig")
    print_dist(split_orig.train, "ORIG train label dist (top)")
    print_dist(split_orig.test,  "ORIG test  label dist (top)")

    # --- DEDUP (dedup.csv) ---
    dedup_path = HERE / "dedup.csv"
    if dedup_path.exists():
        dedup = load_csv(dedup_path)
        print(f"\nDEDUP loaded: {len(dedup)} rows, {dedup['label'].nunique()} labels")
        split_dedup = stratified_split_safe(dedup, test_size=TEST_SIZE, seed=SEED)
        save_split(split_dedup, "dedup")
        print_dist(split_dedup.train, "DEDUP train label dist (top)")
        print_dist(split_dedup.test,  "DEDUP test  label dist (top)")
    else:
        print("\nВнимание: dedup.csv не найден. Пропускаю сплит DEDUP.")

    print("\nSaved files:")
    for name in ["train_orig.csv","test_orig.csv","train_dedup.csv","test_dedup.csv"]:
        p = HERE / name
        if p.exists():
            print("  -", p)

if __name__ == "__main__":
    main()
