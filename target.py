#THIS IS THE MOST APT CODE
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Load the dataset
# Replace 'your_dataset.csv' with the path to your dataset file
dataset = pd.read_csv('/content/train.csv')

# Function to determine problem type based on the target variable's statistical properties
def determine_problem_type(target_column):
    # Check if the target column is numeric
    if pd.api.types.is_numeric_dtype(target_column):
        # Check if the target variable is continuous or categorical based on the number of unique values and range
        num_unique_values = target_column.nunique()
        data_range = target_column.max() - target_column.min()

        # Adjust thresholds based on empirical testing or domain knowledge
        if num_unique_values < 10:
            return 'Classification'
        elif data_range < 0.1 * num_unique_values:
            return 'Classification'
        else:
            return 'Regression'
    else:
        # If the target column is not numeric, assume it's a classification problem
        return 'Classification'

# Assuming the target column is the last column in the dataset
target_column = dataset.iloc[:, -1]

# Determine the problem type
problem_type = determine_problem_type(target_column)

print(f"The identified machine learning problem type is: {problem_type}")
