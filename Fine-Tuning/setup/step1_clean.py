import pandas as pd
import re
import unicodedata

# ------------------------------------------------------------------ #
# CONFIGURATION
# ------------------------------------------------------------------ #
INPUT_FILE = r"C:\Users\Kareem\Desktop\GP\Fine-tuning\diacritized_train_set.xlsx"       
OUTPUT_FILE = "cleaned_dialect.jsonl"
MIN_WORDS = 5

# Sources to exclude entirely
EXCLUDED_SOURCES = {"Shami Corpora", "DART-Dataset"}

# Priority order for sampling if we exceed 40,000 rows
SOURCE_PRIORITY = ["Facebook", "Instagram", "Twitter", "Youtube", "Movie"]

# ------------------------------------------------------------------ #
# LOAD
# ------------------------------------------------------------------ #
print("Loading Excel file...")
df = pd.read_excel(INPUT_FILE)
print(f"Loaded {len(df):,} rows")
print(f"Columns: {list(df.columns)}")

# ------------------------------------------------------------------ #
# STEP 1 - Keep only the Text column and Source column for filtering
# ------------------------------------------------------------------ #
df = df[["Source", "Text"]].copy()

# ------------------------------------------------------------------ #
# STEP 2 - Exclude deprioritized sources
# ------------------------------------------------------------------ #
before = len(df)
df = df[~df["Source"].isin(EXCLUDED_SOURCES)]
print(f"After excluding Shami Corpora + DART-Dataset: {len(df):,} rows (removed {before - len(df):,})")

# ------------------------------------------------------------------ #
# STEP 3 - Drop nulls in Text
# ------------------------------------------------------------------ #
df = df.dropna(subset=["Text"])
df["Text"] = df["Text"].astype(str).str.strip()

# ------------------------------------------------------------------ #
# STEP 4 - Remove rows containing any English characters (a-z, A-Z)
# ------------------------------------------------------------------ #
before = len(df)
df = df[~df["Text"].str.contains(r"[a-zA-Z]", regex=True)]
print(f"After removing English characters: {len(df):,} rows (removed {before - len(df):,})")

# ------------------------------------------------------------------ #
# STEP 5 - Remove rows with URLs
# ------------------------------------------------------------------ #
before = len(df)
url_pattern = r"http[s]?://\S+|www\.\S+"
df = df[~df["Text"].str.contains(url_pattern, regex=True)]
print(f"After removing URLs: {len(df):,} rows (removed {before - len(df):,})")

# ------------------------------------------------------------------ #
# STEP 6 - Remove rows with hashtags or usernames
# ------------------------------------------------------------------ #
before = len(df)
df = df[~df["Text"].str.contains(r"[#@]", regex=True)]
print(f"After removing hashtags/usernames: {len(df):,} rows (removed {before - len(df):,})")

# ------------------------------------------------------------------ #
# STEP 7 - Remove rows shorter than 5 words
# ------------------------------------------------------------------ #
before = len(df)
df = df[df["Text"].str.split().str.len() >= MIN_WORDS]
print(f"After removing rows shorter than {MIN_WORDS} words: {len(df):,} rows (removed {before - len(df):,})")

# ------------------------------------------------------------------ #
# STEP 8 - Remove exact duplicate Text rows
# ------------------------------------------------------------------ #
before = len(df)
df = df.drop_duplicates(subset=["Text"])
print(f"After removing duplicates: {len(df):,} rows (removed {before - len(df):,})")

# ------------------------------------------------------------------ #
# STEP 9 - Remove near-empty or whitespace-only rows after stripping
# ------------------------------------------------------------------ #
df = df[df["Text"].str.strip().str.len() > 0]

# ------------------------------------------------------------------ #
# STEP 10 - Priority-based sampling to stay within 30k-40k target
# ------------------------------------------------------------------ #
print(f"\nRow counts by source after cleaning:")
print(df["Source"].value_counts())

TARGET_MAX = 40000
TARGET_MIN = 30000

if len(df) > TARGET_MAX:
    print(f"\nDataset has {len(df):,} rows, sampling down to {TARGET_MAX:,} using priority order...")
    sampled_parts = []
    remaining_budget = TARGET_MAX

    for source in SOURCE_PRIORITY:
        source_df = df[df["Source"] == source]
        take = min(len(source_df), remaining_budget)
        sampled_parts.append(source_df.sample(n=take, random_state=42))
        remaining_budget -= take
        print(f"  {source}: took {take:,} rows")
        if remaining_budget <= 0:
            break

    df = pd.concat(sampled_parts).reset_index(drop=True)
    print(f"Final sampled dataset: {len(df):,} rows")
else:
    print(f"\nDataset is within target range at {len(df):,} rows. No sampling needed.")

# ------------------------------------------------------------------ #
# STEP 11 - Export to JSONL in Qwen2.5 chat format
# ------------------------------------------------------------------ #
import json

SYSTEM_MESSAGE = "تحدث باللهجة الأردنية العامية دائماً"

print(f"\nExporting to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "assistant", "content": row["Text"]}
            ]
        }
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"Done. Exported {len(df):,} examples to {OUTPUT_FILE}")