#!/usr/bin/env python3

import os
import requests
import sys

OUTPUT_DIR = "huggingface_dataset"
COMBINED_URL = "https://huggingface.co/datasets/magnea/fake-news-formated/resolve/main/combined_data.csv"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined_data.csv")

print(f"Telechargement {os.path.basename(OUTPUT_FILE)}")

response = requests.get(COMBINED_URL, stream=True)
response.raise_for_status()  

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
print(f"Fini de telecharger {os.path.basename(OUTPUT_FILE)}") 