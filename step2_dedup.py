# step2_dedup_safe.py
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
IN_CSV  = HERE / "normalized.csv"
OUT_CSV = HERE / "dedup.csv"

# --- как решаем ничьи между метками для одного текста ---
# варианты: "first" (первая встреченная), "lexi" (лексикографически), "prefer_inform"
TIE_BREAK = "prefer_inform"

def tie_break(candidates):
    if TIE_BREAK == "prefer_inform":
        # если среди кандидатов есть 'inform', выбираем её, иначе берём лексикографически первую
        return "inform" if "inform" in candidates else sorted(candidates)[0]
    elif TIE_BREAK == "first":
        return candidates[0]
    else:  # "lexi"
        return sorted(candidates)[0]

def majority_label_safe(s: pd.Series) -> str | None:
    # убираем NaN и пустые строки
    s = s.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return None
    vc = s.value_counts()
    top = vc.max()
    candidates = vc[vc == top].index.tolist()
    return tie_break(candidates)

def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Не найден {IN_CSV}. Сначала запусти шаг 1 (нормализация).")

    # читаем как строки, чтобы не потерять ничего
    df = pd.read_csv(IN_CSV, dtype={"label": "string", "text": "string"})

    # приводим к чистому виду: убираем полностью пустые тексты
    before = len(df)
    df["label"] = df["label"].fillna("null").str.strip()
    df["text"]  = df["text"].fillna("").str.strip()
    df = df[df["text"] != ""].copy()
    after_drop_empty = len(df)

    # считаем, сколько осталось NaN/пустых меток (на всякий случай)
    n_bad_labels = int((df["label"].isna() | (df["label"] == "")).sum())
    if n_bad_labels:
        # заполняем остатки безопасным классом
        df.loc[df["label"].isna() | (df["label"] == ""), "label"] = "null"

    # дедуп по ТЕКСТУ -> оставляем одну строку с мажоритарной меткой
    dedup = (
        df.groupby("text", dropna=False)["label"]
          .apply(majority_label_safe)
          .reset_index()
          .rename(columns={"label": "label"})
    )
    # если вдруг остались None (все метки были пустыми), выкинем такие строки
    dedup = dedup.dropna(subset=["label"]).copy()
    dedup["label"] = dedup["label"].astype(str)

    # отчёты
    print(f"Прочитано строк: {before}")
    print(f"После удаления пустых текстов: {after_drop_empty}")
    print(f"Пустых/NaN меток до заполнения: {n_bad_labels}")
    print(f"Итог после дедупликации (по тексту): {len(dedup)}")
    removed = after_drop_empty - len(dedup)
    print(f"Удалено дубликатов: {removed}")

    print("\nТоп-10 меток ДО дедупликации:")
    print(df["label"].value_counts().head(10))
    print("\nТоп-10 меток ПОСЛЕ дедупликации:")
    print(dedup["label"].value_counts().head(10))

    dedup.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nСохранено: {OUT_CSV}")

    # мини-превью
    print("\nПервые 10 строк dedup.csv:")
    print(dedup.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
