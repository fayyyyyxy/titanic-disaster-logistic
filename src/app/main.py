# src/app/main.py
import sys, platform
print("Titanic starter running ✅")
print("Python:", platform.python_version())
print("Args:", sys.argv[1:] or "(none)")

import pandas as pd
train_df = pd.read_csv("src/data/train.csv")
print("Shape:", train_df.shape)
print("Columns:", list(train_df.columns))
