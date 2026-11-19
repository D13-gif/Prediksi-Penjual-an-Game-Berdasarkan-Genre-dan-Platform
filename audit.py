import pandas as pd
import numpy as np

df = pd.read_csv('vgsales.csv')
print("=" * 60)
print("DATASET AUDIT")
print("=" * 60)

print("\n1. SHAPE:")
print(df.shape)

print("\n2. COLUMNS:")
print(df.columns.tolist())

print("\n3. DTYPES:")
print(df.dtypes)

print("\n4. MISSING VALUES:")
print(df.isnull().sum())

print("\n5. DUPLICATES:")
print(df.duplicated().sum())

print("\n6. HEAD:")
print(df.head())

print("\n7. DESCRIBE (numeric):")
print(df.describe())

print("\n8. UNIQUE VALUES (categorical):")
for col in ['Platform', 'Genre', 'Publisher']:
    print(f"  {col}: {df[col].nunique()} unique")
    print(f"    top 5: {df[col].value_counts().head(5).tolist()}")

print("\n9. YEAR RANGE:")
print(f"  min={df['Year'].min()}, max={df['Year'].max()}")
print(f"  nulls={df['Year'].isnull().sum()}")

print("\n10. SALES COLUMNS:")
for col in ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']:
    print(f"  {col}: min={df[col].min()}, max={df[col].max():.2f}, mean={df[col].mean():.2f}, median={df[col].median():.2f}")

print("\n11. ZERO SALES:")
print(f"  Global_Sales == 0: {(df['Global_Sales']==0).sum()}")
print(f"  NA_Sales == 0: {(df['NA_Sales']==0).sum()}")

print("\n12. NULL PUBLISHER:")
print(df['Publisher'].isnull().sum())

print("\n13. TAIL:")
print(df.tail())