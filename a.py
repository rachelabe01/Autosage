import streamlit as st
import numpy as np
import librosa
import os
import tempfile
import zipfile
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

def calculate_mfccs(signal, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def preprocess_audio(audio_path):
    signal, sr = librosa.load(audio_path, sr=None)
    mfccs = calculate_mfccs(signal, sr)
    return mfccs

def process_zip(zip_path):
    temp_dir = tempfile.mkdtemp()
    mfccs_list = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                mfccs = preprocess_audio(file_path)
                mfccs_list.append(mfccs)
    shutil.rmtree(temp_dir)
    return np.array(mfccs_list)

def analyze_and_cluster_datasets(features_dict):
    best_score = -1
    best_model = None
    best_model_name = ""
    silhouette_threshold = 0.5  # Adjust based on your domain knowledge and dataset characteristics

    for name, features in features_dict.items():
        if len(features) == 0:
            st.write(f"Dataset {name} is empty or contains invalid data.")
            continue

        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)

        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(normalized_features)

        clustering = AgglomerativeClustering(n_clusters=2)
        cluster_labels = clustering.fit_predict(pca_results)

        score = silhouette_score(pca_results, cluster_labels)
        task_type = "Classification" if score >= silhouette_threshold else "Clustering"
        st.write(f"Dataset: {name}, Silhouette Score: {score:.2f}, Suggested Task: {task_type}")

        if score > best_score:
            best_score = score
            if best_model is not None:
                os.remove(best_model_name)  # Clean up previous best model
            best_model = clustering
            best_model_name = f'{name}_model.joblib'
            joblib.dump(best_model, best_model_name)

    return best_model_name

st.title('Audio Dataset Analyzer with MFCC, PCA, and Agglomerative Clustering')

uploaded_files = st.file_uploader("Upload Zip files containing audio datasets", accept_multiple_files=True, type=['zip'])

if uploaded_files:
    dataset_features = {}
    for uploaded_file in uploaded_files:
        with st.spinner(f'Processing {uploaded_file.name}...'):
            features = process_zip(uploaded_file)
            if features.size > 0:
                dataset_features[uploaded_file.name] = features

    if dataset_features:
        best_model_name = analyze_and_cluster_datasets(dataset_features)
        if best_model_name:
            with open(best_model_name, "rb") as f:
                st.download_button(f"Download Best Model ({best_model_name})", f, file_name=best_model_name)
        else:
            st.write("No valid datasets were processed or no clear best dataset was found.")
else:
    st.write("Please upload one or more zip files containing your audio datasets in WAV format.")
