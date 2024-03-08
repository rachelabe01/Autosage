import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import precision_score, f1_score, mean_squared_error, r2_score
import pickle
import base64
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

def find_potential_target_column(df):
    df_numeric = df.select_dtypes(include=['number'])  # Select only numeric columns
    df_numeric.dropna(axis=1, how='all', inplace=True)  # Drop columns with all NaN values

    if len(df_numeric.columns) > 2000:
        return find_potential_target_column_using_silhouette_score(df_numeric)
    else:
        return find_potential_target_column_using_correlation(df_numeric)

def find_potential_target_column_using_silhouette_score(df_numeric):
    # Fill any missing values
    df_numeric.fillna(df_numeric.mean(), inplace=True)

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

def find_potential_target_column_using_correlation(df_numeric):
    # Fill any missing values
    df_numeric.fillna(df_numeric.mean(), inplace=True)

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

def evaluate_classification(df):
    X = df.drop(columns=[df.columns[-1]])
    y = df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, precision, f1, model

def evaluate_regression(df):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    X = df[numeric_columns].drop(columns=[df.columns[-1]])  # Exclude target column
    y = df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, model

def compare_datasets(dataset_info):
    sorted_datasets = sorted(dataset_info, key=lambda x: (x['accuracy'] if x['problem_type'] == 'Classification' else -x['mse']), reverse=True)
    return sorted_datasets[:3]

def main():
    st.title("Dataset Comparison App")
    st.write("Upload multiple datasets to identify their problem types, evaluate them, and rank them.")

    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

    dataset_info = []
    if uploaded_files:
        for file in uploaded_files:
            file_name = file.name
            df = pd.read_csv(file)
            problem_type = identify_problem_type(df)
            st.write(f"**{file_name}**: {problem_type}")
            st.write(df)

            df = df.select_dtypes(include=np.number)  # Drop categorical columns

            if problem_type == 'Classification':
                accuracy, precision, f1, model = evaluate_classification(df)
                metrics = {'accuracy': accuracy, 'precision_score': precision, 'f1_score': f1}
            elif problem_type == 'Regression':
                mse, r2, model = evaluate_regression(df)
                metrics = {'mse': mse, 'r2_score': r2}
            else:
                st.warning(f"Skipping evaluation for dataset {file_name} as its problem type is unknown.")
                continue

            dataset_info.append({'name': file_name, 'problem_type': problem_type, **metrics, 'model': model})

    if st.button("Compare"):
        if len(dataset_info) < 2:
            st.error("Please upload at least 2 datasets to compare.")
            return

        top_datasets = compare_datasets(dataset_info)
        st.success("The top 3 datasets for your project are:")
        for idx, dataset in enumerate(top_datasets):
            st.write(f"{idx + 1}. Name: {dataset['name']}, Problem Type: {dataset['problem_type']}")
            if dataset['problem_type'] == 'Classification':
                st.write(f"   Accuracy: {dataset['accuracy']}, Precision Score: {dataset['precision_score']}, F1 Score: {dataset['f1_score']}")
            elif dataset['problem_type'] == 'Regression':
                st.write(f"   Mean Squared Error: {dataset['mse']}, R2 Score: {dataset['r2_score']}")

        # Select top-ranked dataset
        top_dataset = top_datasets[0]
        st.subheader(f"Modeling Top Ranked Dataset: {top_dataset['name']}")
        st.write(f"Problem Type: {top_dataset['problem_type']}")

        # Display model accuracy
        if top_dataset['problem_type'] == 'Classification':
            st.write(f"Model Accuracy: {top_dataset['accuracy']}, Precision Score: {top_dataset['precision_score']}, F1 Score: {top_dataset['f1_score']}")
        elif top_dataset['problem_type'] == 'Regression':
            st.write(f"Model Mean Squared Error: {top_dataset['mse']}, R2 Score: {top_dataset['r2_score']}")

        # Save model
        model_name = top_dataset['name'].split('.')[0] + '_model.pkl'
        with open(model_name, 'wb') as f:
            pickle.dump(top_dataset['model'], f)

        # Display download link
        st.write("Do you want to download the trained model?")
        with open(model_name, "rb") as f:
            model_bytes = f.read()
            st.download_button(
                label="Download Model",
                data=model_bytes,
                file_name=model_name,
                mime="application/octet-stream"
            )

if __name__ == "__main__":
    main()
