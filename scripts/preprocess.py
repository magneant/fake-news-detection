#!/usr/bin/env python3

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


COMBINED_DIR = "huggingface_dataset"
COMBINED_FILE = os.path.join(COMBINED_DIR, "combined_data.csv")
TRAIN_DIR = "trainingData"
TEST_DIR = "testingData"
TRAIN_FILE = os.path.join(TRAIN_DIR, "train.csv")
TEST_FILE = os.path.join(TEST_DIR, "test.csv")
TEST_RATIO = 0.1
RANDOM_STATE = 42


os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

df = pd.read_csv(COMBINED_FILE, low_memory=False, on_bad_lines='warn')

df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=['object']):
    df[col] = df[col].str.replace('\r', '', regex=True).str.strip()
df.dropna(how='all', inplace=True)
print(f"{len(df)} ont été nettoyés")

if len(df) == 0:
    print("nothing to do", file=sys.stderr)
    sys.exit(1)
train_df, test_df = train_test_split(
    df, test_size=TEST_RATIO, random_state=RANDOM_STATE, shuffle=True
)

train_df.to_csv(TRAIN_FILE, index=False, encoding='utf-8')
test_df.to_csv(TEST_FILE, index=False, encoding='utf-8')

print(f"Sauvegarder {len(train_df)} training rows to {TRAIN_FILE}")
print(f"Sauvegarder {len(test_df)} test rows to {TEST_FILE}") 