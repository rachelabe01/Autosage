import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Function to handle missing values
def handle_missing_values(dataset):
    missing_values = dataset.isnull().sum()
    if missing_values.any():
        imputer = SimpleImputer(strategy='mean')
        dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
    return dataset

# Function to determine problem type
def determine_problem_type(target_column):
    if pd.api.types.is_numeric_dtype(target_column):
        num_unique_values = target_column.nunique()
        data_range = target_column.max() - target_column.min()
        if num_unique_values < 10 or data_range < 0.1 * num_unique_values:
            return 'Classification'
        else:
            return 'Regression'

# Main function
def main():
    st.title('Machine Learning Problem Type Detection and Analysis')

    # Upload files
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=['csv'])

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"### {uploaded_file.name}")
            dataset = pd.read_csv(uploaded_file)

            # Label encoding for categorical columns
            categorical_columns = dataset.select_dtypes(include=['object']).columns
            label_encoder = LabelEncoder()
            for column in categorical_columns:
                dataset[column] = label_encoder.fit_transform(dataset[column])

            # Handling missing values
            dataset = handle_missing_values(dataset)

            # Determine problem type
            target_column = dataset.iloc[:, -1]
            problem_type = determine_problem_type(target_column)
            st.write(f"The identified machine learning problem type is: {problem_type}")

if __name__ == '__main__':
    main()
