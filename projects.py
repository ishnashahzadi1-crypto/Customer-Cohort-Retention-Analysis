import pandas as pd

df = pd.read_csv("olist_6_files_merged.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())