import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.manifold import TSNE
import tempfile
import os
import base64

st.title('Dataset Classifier or Regressor Selector')

def preprocess_dataset(dataset):
    """
    Preprocess the dataset, determine problem type, and encode target.
    """
    target = dataset.iloc[:, -1]
    features = dataset.iloc[:, :-1]
    
    # Assume problem type based on target variable type and uniqueness
    problem_type = 'Classification' if target.dtype == 'object' or len(target.unique()) < 20 else 'Regression'
    
    # Normalize numeric features and encode categorical features
    scaler = MinMaxScaler()
    features = pd.get_dummies(features)
    numeric_features = features.select_dtypes(include=['float64', 'int64'])
    features[numeric_features.columns] = scaler.fit_transform(numeric_features)
    
    if problem_type == 'Classification':
        encoder = LabelEncoder()
        target = encoder.fit_transform(target)
    
    return features, target, problem_type

def build_and_train_model(X, y, problem_type):
    """
    Build and train a TensorFlow model based on problem type.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1 if problem_type == 'Regression' else len(np.unique(y)), activation='linear' if problem_type == 'Regression' else 'softmax')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error' if problem_type == 'Regression' else 'sparse_categorical_crossentropy', metrics=['mse' if problem_type == 'Regression' else 'accuracy'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    predictions = model.predict(X_test)
    if problem_type == 'Classification':
        predictions = np.argmax(predictions, axis=1)
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted')
        }
    else:
        metrics = {'mse': mean_squared_error(y_test, predictions)}
    
    return model, metrics

def main():
    uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type=['csv'])
    if uploaded_files:
        best_score = -np.inf
        best_dataset_info = {}
        for uploaded_file in uploaded_files:
            dataset = pd.read_csv(uploaded_file)
            features, target, problem_type = preprocess_dataset(dataset)
            model, metrics = build_and_train_model(features.values, target, problem_type)
            
            st.write(f"Dataset: {uploaded_file.name}")
            st.write(dataset.head())
            st.write(f"Problem Type: {problem_type}, Metrics: {metrics}")
            
            score = metrics['accuracy'] if problem_type == 'Classification' else -metrics['mse']
            if score > best_score:
                best_score = score
                best_dataset_info = {
                    'dataset': dataset,
                    'metrics': metrics,
                    'problem_type': problem_type,
                    'model': model
                }

        if best_dataset_info:
            st.write(f"Best Dataset: {uploaded_file.name}, Problem Type: {best_dataset_info['problem_type']}, Best Score: {best_score}")
            st.write(best_dataset_info['metrics'])
            
            # Offer download of best dataset
            #tmp_download_link = download_link(best_dataset_info['dataset'], "best_dataset.csv", "Download best dataset")
            #st.markdown(tmp_download_link, unsafe_allow_html=True)
            
            # Save and offer download of best model
            model_file_path = save_model(best_dataset_info['model'])
            st.download_button(label="Download Best Model", data=open(model_file_path, "rb"), file_name="best_model.h5")

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def save_model(model):
    """
    Save the model to a temporary file and return the file path.
    """
    _, model_path = tempfile.mkstemp(suffix=".h5")
    model.save(model_path)
    return model_path

if __name__ == "__main__":
    main()
