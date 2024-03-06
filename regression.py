#C:/Users/rachel/Downloads/CarPrice_Assignment.csv
import pandas as pd

# Load your dataset
df = pd.read_csv('C:/Users/rachel/Downloads/housing (7).csv')

# Convert categorical columns to numerical using label encoding
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = pd.factorize(df[col])[0]

# Find the column with the highest mean value
highest_mean_column = df.mean().idxmax()

# Find the column with the highest maximum value
highest_max_column = df.max().idxmax()

print("Column with the highest mean value:", highest_mean_column)
print("Column with the highest maximum value:", highest_max_column)

