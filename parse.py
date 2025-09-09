# step1_normalize.py  (parse.py)
from pathlib import Path
import re
import pandas as pd

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()              # схлопываем пробелы
    s = re.sub(r"\b\d{3,}\b", "<num>", s)           # длинные числа -> <num>
    s = re.sub(r"\b(?:tv_noise|noise|sil)\b", "<noise>", s)  # шум -> <noise>
    return s

def load_and_normalize(data_path: Path) -> pd.DataFrame:
    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)          # label + (utterance)
            label = parts[0]
            text  = parts[1] if len(parts) == 2 else ""
            rows.append((label, normalize_text(text)))
    return pd.DataFrame(rows, columns=["label", "text"])

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_path  = script_dir / "dialog_acts.dat"     # ссылка на файл из той же директории
    if not data_path.exists():
        raise FileNotFoundError(f"Не найден {data_path}. Положи dialog_acts.dat рядом со скриптом.")

    df = load_and_normalize(data_path)
    out_csv = script_dir / "normalized.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # быстрый отчёт
    print(f"Saved {len(df)} rows -> {out_csv}")
    print(f"Unique labels: {df['label'].nunique()}")
    print("\nTop labels:")
    print(df["label"].value_counts().head(15))
    print("\nSample:\n", df.head(10).to_string(index=False))