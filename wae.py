# import pandas as pd

# # Load your dataset
# # Replace 'your_dataset.csv' with the path to your dataset
# data = pd.read_csv('C:/Users/rachel/OneDrive/Desktop/trying/tips.csv')

# # One-hot encode categorical variables
# data_encoded = pd.get_dummies(data)

# # Calculate correlation matrix
# correlation_matrix = data_encoded.corr()

# # Set the threshold for strong correlation
# threshold = 0.5  # You can adjust this threshold based on your needs

# # Find strongly correlated variables
# strongly_correlated_variables = set()
# num_variables = correlation_matrix.shape[0]

# for i in range(num_variables):
#     for j in range(i+1, num_variables):
#         if abs(correlation_matrix.iloc[i, j]) > threshold:
#             strongly_correlated_variables.add(correlation_matrix.columns[i])
#             strongly_correlated_variables.add(correlation_matrix.columns[j])

# # Print strongly correlated variables
# print("Strongly correlated variables:")
# print(strongly_correlated_variables)

# import pandas as pd
# import numpy as np

# # Load your unknown dataset
# # Replace 'your_unknown_dataset.csv' with the path to your dataset
# data = pd.read_csv('C:/Users/rachel/Downloads/Student_Marks.csv')

# # Compute Spearman correlation coefficients for all pairs of variables
# spearman_corr = data.corr(method='spearman')

# # Exclude correlations with the same variable (diagonal elements)
# spearman_corr = spearman_corr.mask(np.tril(np.ones(spearman_corr.shape)).astype(bool))
# # Extract absolute correlation coefficients for each variable
# abs_spearman_corr = spearman_corr.abs()

# # Identify potential dependent variables based on highest absolute correlation coefficients
# potential_dependent_variables = abs_spearman_corr.idxmax()

# # Print potential dependent variables
# print("Potential dependent variables:")
# print(potential_dependent_variables)
#C:/Users/rachel/OneDrive/Desktop/trying/diabetes (3).csv
#C:/Users/rachel/OneDrive/Desktoptrying/wine (1).csv
import pandas as pd
#C:/Users/rachel/Downloads/banking_loanapproval.csv
# Load your unknown dataset
# Replace 'your_unknown_dataset.csv' with the path to your dataset
data = pd.read_csv('C:/Users/rachel/Downloads/banking_loanapproval.csv')

# Compute Spearman correlation coefficients for all pairs of variables
spearman_corr = data.corr(method='spearman')

# Exclude correlations with the same variable (diagonal elements)
spearman_corr = spearman_corr.mask(spearman_corr == 1.0)

# Identify potential target variables based on their correlation with other variables
potential_target_columns = []

for column in spearman_corr.columns:
    if any(spearman_corr[column].abs() > 0.7):  # You can adjust the threshold as needed
        potential_target_columns.append(column)

# Print potential target columns
print("Potential target columns based on Spearman correlation:")
print(potential_target_columns)


