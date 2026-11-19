import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('notebooks/figures', exist_ok=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

df = pd.read_csv('vgsales.csv')
print('Shape:', df.shape)

print('\nMissing values:')
print(df.isnull().sum())
print('\nDuplicated:', df.duplicated().sum())
print('\nDtypes:')
print(df.dtypes)

df['Sales_Category'] = pd.cut(
    df['Global_Sales'],
    bins=[-np.inf, 0.5, 2.0, np.inf],
    labels=['Rendah', 'Sedang', 'Tinggi']
)
print('\nSales_Category distribution:')
print(df['Sales_Category'].value_counts())
print('\nPercentages:')
print(df['Sales_Category'].value_counts(normalize=True) * 100)

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Sales_Category', palette='viridis')
plt.title('Distribution of Sales_Category')
plt.tight_layout()
plt.savefig('notebooks/figures/sales_category_dist.png')
plt.close()

print('\nGenre distribution:')
print(df['Genre'].value_counts())
print('\nTop 10 Platforms:')
print(df['Platform'].value_counts().head(10))
print('\nTop 10 Publishers:')
print(df['Publisher'].value_counts().head(10))

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Genre', y='Global_Sales', palette='viridis')
plt.title('Global Sales by Genre')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('notebooks/figures/global_sales_by_genre.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Platform', y='Global_Sales', palette='coolwarm')
plt.title('Global Sales by Platform')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('notebooks/figures/global_sales_by_platform.png')
plt.close()

plt.figure(figsize=(10, 5))
df.groupby(df['Year'].astype('Int64'))['Global_Sales'].mean().plot()
plt.title('Average Global Sales by Year')
plt.xlabel('Year')
plt.ylabel('Avg Global Sales')
plt.tight_layout()
plt.savefig('notebooks/figures/avg_sales_by_year.png')
plt.close()

corr = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Year']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('notebooks/figures/correlation_matrix.png')
plt.close()

print('\nGlobal_Sales stats:')
print(df['Global_Sales'].describe())
print('\nTop 10 best-selling games:')
print(df.nlargest(10, 'Global_Sales')[['Name', 'Platform', 'Genre', 'Global_Sales']])

print('\nNull Publisher:', df['Publisher'].isnull().sum())
print('Null Year:', df['Year'].isnull().sum())
print('\nUnique counts:')
for c in ['Platform', 'Genre', 'Publisher']:
    print(f'  {c}: {df[c].nunique()}')
print('\nYear range:', df['Year'].min(), '-', df['Year'].max())

print('\nEDA completed successfully. Figures saved to notebooks/figures/')