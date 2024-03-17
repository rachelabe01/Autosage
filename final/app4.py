import streamlit as st
from PIL import Image
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tempfile
import os

def app4():
    def process_zip_and_check_type(file, target_size=(64, 64)):
        images = []
        labels = []
        label_map = {}
        directory_set = set()

        with zipfile.ZipFile(file, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    directory_name = file_name.split('/')[0]
                    directory_set.add(directory_name)

                    label = directory_name if len(directory_set) > 1 else 'default'
                    if label not in label_map:
                        label_map[label] = len(label_map)
                    labels.append(label_map[label])

                    with zip_ref.open(file_name) as image_file:
                        img = Image.open(image_file).convert('RGB')
                        img = img.resize(target_size)
                        img_array = np.array(img)
                        images.append(img_array)

        images = np.array(images)
        dataset_type = "clustering" if len(directory_set) <= 1 else "classification"
        labels = np.array(labels) if dataset_type == "classification" else None

        return images, labels, dataset_type, len(directory_set)

    def evaluate_classification(images, labels):
        results = []
        num_classes = len(np.unique(labels))
        labels_cat = to_categorical(labels, num_classes=num_classes)

        X_train, X_val, y_train, y_val = train_test_split(images, labels_cat, test_size=0.2, random_state=42)

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=2)

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        results.append({"Type": "Classification", "Algorithm": "CNN", "Score": val_acc, "Model": model})

        return results, model, val_acc

    def evaluate_clustering(images):
        results = []
        images_reshaped = images.reshape(images.shape[0], -1)
        best_score = -1
        best_model = None
        clustering_algorithms = {
            "KMeans": KMeans(n_clusters=3, random_state=42),
            "Agglomerative": AgglomerativeClustering(n_clusters=3),
        }

        for name, algorithm in clustering_algorithms.items():
            labels = algorithm.fit_predict(images_reshaped)
            score = silhouette_score(images_reshaped, labels) if len(set(labels)) > 1 else 0
            results.append({"Type": "Clustering", "Algorithm": name, "Score": score})
            if score > best_score:
                best_score = score
                best_model = algorithm

        return results, best_model, best_score

    def save_model(model):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            model_path = tmp.name + ".h5"
        model.save(model_path)
        return model_path

    st.title('Multi Image-based Data Model')

    uploaded_files = st.file_uploader("Upload ZIP files containing datasets", accept_multiple_files=True, type="zip")

    if uploaded_files:
        all_results = []
        best_overall_score = -1
        best_model_path = None
        best_dataset_name = ""

        for uploaded_file in uploaded_files:
            st.write(f"Processing dataset: {uploaded_file.name}")
            images, labels, dataset_type, _ = process_zip_and_check_type(uploaded_file, target_size=(64, 64))

            if dataset_type == "classification" and labels is not None:
                results, model, score = evaluate_classification(images, labels)
                if score > best_overall_score:
                    best_overall_score = score
                    best_model_path = save_model(model)
                    best_dataset_name = uploaded_file.name
            elif dataset_type == "clustering":
                results, model, score = evaluate_clustering(images)
                # Clustering models are not saved as they are not typically used for predictions in the same way

            for result in results:
                result['Dataset'] = uploaded_file.name
                all_results.append(result)

        results_df = pd.DataFrame(all_results, columns=["Dataset", "Type", "Algorithm", "Score"])
        
        # Highlight the highest score
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: yellow' if v else '' for v in is_max]
        st.dataframe(results_df.style.apply(highlight_max, subset=['Score']))

        # Download best model
        if best_model_path:
            st.write(f"The best model was from the dataset: {best_dataset_name} with a score of: {best_overall_score}")
            with open(best_model_path, "rb") as file:
                btn = st.download_button(
                    label="Download Best Model",
                    data=file,
                    file_name="best_model.h5",
                    mime="application/octet-stream"
                )
            os.remove(best_model_path)  # Clean up the temporary file
    else:
        st.info("Please upload one or more ZIP files to proceed.")

if __name__ == '__main__':
    app4()
