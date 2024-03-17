import streamlit as st
import numpy as np
import librosa
import os
import tempfile
import zipfile
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score, precision_score
import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def app2():
    # Define functions for audio file processing
    def calculate_mfccs(signal, sr, n_mfcc=13):
        """Calculate MFCCs from audio signal."""
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed

    def preprocess_audio(audio_path):
        """Load and preprocess audio file."""
        signal, sr = librosa.load(audio_path, sr=None)
        return calculate_mfccs(signal, sr)

    def process_zip(zip_path):
        """Extract and process audio files from a zip archive."""
        temp_dir = tempfile.mkdtemp()
        mfccs_list = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    mfccs_list.append(preprocess_audio(file_path))
        shutil.rmtree(temp_dir)
        return np.array(mfccs_list)

    # Define machine learning models
    classification_algorithms = {
        "AdaBoostClassifier": AdaBoostClassifier(n_estimators=100),
        "GaussianNB": GaussianNB(),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100)
    }

    additional_clustering_algorithms = {
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "OPTICS": OPTICS(min_samples=5)
    }

    def analyze_and_cluster_datasets(features_dict):
        """Analyze and determine the best model for datasets."""
        best_model_name = None
        best_model_details = {"score": -float('inf')}
        dataset_info = []

        for name, features in features_dict.items():
            if len(features) == 0:
                continue

            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(features)
            pca = PCA(n_components=min(features.shape[1], 2))
            pca_results = pca.fit_transform(normalized_features)

            dummy_clustering = KMeans(n_clusters=2)
            initial_labels = dummy_clustering.fit_predict(pca_results)
            initial_score = silhouette_score(pca_results, initial_labels)
            task_type = "Classification" if initial_score > 0.5 else "Clustering"

            algorithms = classification_algorithms if task_type == "Classification" else additional_clustering_algorithms

            for algo_name, algorithm in algorithms.items():
                if task_type == "Classification":
                    X_train, X_test, y_train, y_test = train_test_split(pca_results, np.random.randint(2, size=len(pca_results)), test_size=0.3, random_state=42)
                    algorithm.fit(X_train, y_train)
                    y_pred = algorithm.predict(X_test)
                    score = precision_score(y_test, y_pred, average='macro')
                    metric_name = "Precision Score"
                else:
                    cluster_labels = algorithm.fit_predict(pca_results)
                    if len(set(cluster_labels)) < 2 or -5 in cluster_labels:
                        continue
                    score = silhouette_score(pca_results, cluster_labels)
                    metric_name = "Silhouette Score"

                dataset_info.append({
                    "Type": task_type,
                    "Dataset": name,
                    "Algorithm": algo_name,
                    metric_name: score
                })

                if score > best_model_details["score"]:
                    best_model_details = {
                        "dataset": name, "algorithm": algo_name, "score": score, "task_type": task_type
                    }
                    best_model_name = f"{name}_{algo_name}_model.joblib"
                    joblib.dump(algorithm, best_model_name)

        return pd.DataFrame(dataset_info), best_model_name, best_model_details

    # Streamlit UI
    st.title('Automated Audiobased Model')

    uploaded_files = st.file_uploader("Upload ZIP files containing audio datasets", accept_multiple_files=True, type=['zip'])

    if uploaded_files:
        dataset_features = {f.name: process_zip(f) for f in uploaded_files}

        if dataset_features:
            performance_df, best_model_name, best_model_details = analyze_and_cluster_datasets(dataset_features)
            st.dataframe(performance_df)
            if best_model_name:
                st.write(f"Best Model: {best_model_name}")
                st.json(best_model_details)
                with open(best_model_name, "rb") as file:
                    st.download_button("Download Best Model", file, file_name=best_model_name, mime="application/octet-stream")
    else:
        st.write("Please upload one or more ZIP files containing your audio datasets in WAV format.")

# Ensure to call app2() in your main.py or if you want to test this directly, uncomment the line below.
# app2()
if __name__ == '__main__':
    app2()

