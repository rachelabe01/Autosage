import streamlit as st
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import pickle
import base64
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Function to create a download link for a file
def get_download_link(file_content, file_name):
    """
    Generate a download link for the given file content and file name.

    Args:
    - file_content: The binary content of the file to be downloaded.
    - file_name: The name of the file for the download.

    Returns:
    - A string containing an HTML anchor tag with a link to download the file.
    """
    # Encode the file content to base64
    b64 = base64.b64encode(file_content).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return href

# Function to preprocess data
def preprocess_data(df):
    # Drop categorical columns
    df_numeric = df.select_dtypes(include=np.number)

    # Handling missing values
    imputer = SimpleImputer(strategy='mean')
    df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

    # Scale the features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    return df_scaled

# Function to evaluate the top dataset
def evaluate_top_dataset(df_scaled):
    X = df_scaled.drop(columns=[df_scaled.columns[-1]])
    y = df_scaled[df_scaled.columns[-1]]

    if identify_problem_type(df_scaled) == 'Classification':
        # Convert target variable to categorical for classification
        y = y.astype(int).astype(str)

        model = HistGradientBoostingClassifier()
    else:
        model = HistGradientBoostingRegressor()

    model.fit(X, y)
    return model

# Function to evaluate a single dataset
def evaluate_dataset(file):
    df = pd.read_csv(file, nrows=1000)  # Read only the first 1000 rows for efficiency

    st.write(f"**{file.name}**")
    st.write(df.head())

    df_scaled = preprocess_data(df)

    problem_type = identify_problem_type(df_scaled)
    st.write(f"Problem Type: {problem_type}")

    model = evaluate_top_dataset(df_scaled)
    dataset_info = {'name': file.name, 'problem_type': problem_type, 'model': model, 'df_scaled': df_scaled}

    return {'name': file.name, 'problem_type': problem_type, 'model': model, 'df_scaled': df_scaled}


# Function to identify problem type
def identify_problem_type(df):
    # Find the potential target column
    potential_target_column = find_potential_target_column(df)

    # Count the number of unique values in the potential target column
    unique_values_count = df[potential_target_column].nunique()

    # Check if the unique values count is less than or equal to 3
    if unique_values_count <= 3:
        return 'Classification'
    else:
        return 'Regression'

# Function to find potential target column using correlation
def find_potential_target_column_using_correlation(df_numeric):
    # Fill missing values
    imputer = SimpleImputer(strategy='mean')
    df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_numeric)

    # Perform t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(scaled_features)

    # Compute correlation with t-SNE components for each column
    correlation_with_tsne = []
    for col in df_numeric.columns:
        correlation = np.corrcoef(df_numeric[col], tsne_results[:, 0])[0, 1]
        correlation_with_tsne.append(abs(correlation))

    # Find the column with the highest correlation with t-SNE components
    most_potential_target_column = df_numeric.columns[np.argmax(correlation_with_tsne)]
    return most_potential_target_column

# Function to find potential target column using silhouette score
def find_potential_target_column_using_silhouette_score(df_numeric):
    # Fill missing values
    imputer = SimpleImputer(strategy='mean')
    df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_numeric)

    # Perform t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(scaled_features)

    # Compute silhouette score for each column
    silhouette_scores = []
    for col in df_numeric.columns:
        target = df_numeric[col]
        silhouette_scores.append(silhouette_score(tsne_results, target))

    # Find the column with the highest silhouette score
    most_potential_target_column = df_numeric.columns[np.argmax(silhouette_scores)]
    return most_potential_target_column

# Function to find potential target column
def find_potential_target_column(df):
    df_numeric = df.select_dtypes(include=['number'])  # Select only numeric columns
    df_numeric.dropna(axis=1, how='all', inplace=True)  # Drop columns with all NaN values

    if len(df_numeric.columns) > 2000:
        return find_potential_target_column_using_silhouette_score(df_numeric)
    else:
        return find_potential_target_column_using_correlation(df_numeric)

# Function to rank datasets based on t-SNE correlation
def rank_datasets(results):
    ranked_datasets = []

    for result in results:
        df_scaled = result['df_scaled']
        potential_target_column = find_potential_target_column(df_scaled)

        # Perform t-SNE to reduce dimensions to 2D
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(df_scaled.drop(columns=[potential_target_column]))

        # Compute correlation with t-SNE components for the potential target column
        correlation = np.corrcoef(df_scaled[potential_target_column], tsne_results[:, 0])[0, 1]

        ranked_datasets.append({'name': result['name'], 'correlation': correlation})

    # Sort the datasets based on correlation
    ranked_datasets.sort(key=lambda x: abs(x['correlation']), reverse=True)

    return ranked_datasets

# Main function
def main():
    # Set page title and layout
    st.set_page_config(page_title="Dataset Modeling and Comparison App", layout="wide")

    # Main title
    st.title("Dataset Modeling and Comparison App")

    # Sidebar
    st.sidebar.title("Settings")

    # File upload
    uploaded_files = st.sidebar.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

    if uploaded_files:
        dataset_info = []

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(evaluate_dataset, uploaded_files))

            # Rank datasets based on t-SNE correlation
            ranked_datasets = rank_datasets(results)

            # Display ranked datasets
            st.subheader("Ranked Datasets based on t-SNE Correlation")
            for i, dataset in enumerate(ranked_datasets):
                st.write(f"{i+1}. {dataset['name']} - Correlation: {dataset['correlation']}")

            # Display dataset comparison
            st.subheader("Dataset Comparison")
            df_comparison = pd.DataFrame(ranked_datasets)
            st.write(df_comparison)

        # Display performance metrics for each dataset...
        
if __name__ == "__main__":
    main()
