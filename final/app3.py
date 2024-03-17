import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import zipfile
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tempfile
import os
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def app3():
    # Define the CNN model architecture
    def create_cnn_model(input_shape=(64, 64, 3), num_classes=2):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Function to process images from the uploaded ZIP file
    def process_zip(file, target_size=(64, 64)):
        images = []
        labels = []
        label_map = {}

        with zipfile.ZipFile(file, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    label = file_name.split('/')[0]  # Assuming the folder name is the label
                    if label not in label_map:
                        label_map[label] = len(label_map)
                    labels.append(label_map[label])
                    
                    with zip_ref.open(file_name) as image_file:
                        img = Image.open(image_file).convert('RGB')
                        img = img.resize(target_size)
                        img_array = np.array(img)
                        images.append(img_array)
        
        images = np.array(images)
        labels = np.array(labels)
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_map))
        
        return images, labels, label_map

    def calculate_silhouette_scores(images):
        tsne = TSNE(n_components=2, random_state=42)
        images_tsne = tsne.fit_transform(images.reshape(images.shape[0], -1))
        kmeans = KMeans(n_clusters=2, random_state=42).fit(images_tsne)
        score = silhouette_score(images_tsne, kmeans.labels_)
        return score

    st.title('Automated Imagebased Model')

    uploaded_files = st.file_uploader("Upload ZIP files containing datasets", accept_multiple_files=True, type="zip")

    if uploaded_files:
        dataset_scores = {}
        for uploaded_file in uploaded_files:
            st.write(f"Processing dataset: {uploaded_file.name}")
            images, labels, label_map = process_zip(uploaded_file, target_size=(64, 64))
            score = calculate_silhouette_scores(images)
            dataset_scores[uploaded_file.name] = score

        if dataset_scores:
            # Decision based on silhouette score threshold
            threshold = 0.5  # Adjust this threshold as needed
            best_dataset_name = max(dataset_scores, key=dataset_scores.get)
            best_score = dataset_scores[best_dataset_name]
            decision = "clustering" if best_score >= threshold else "classification"
            
            st.write(f"The best dataset for {decision} is {best_dataset_name} with a silhouette score of {best_score:.2f}.")
            
            for dataset_name, score in dataset_scores.items():
                decision_here = "better suited for clustering" if score >= threshold else "potentially better suited for classification"
                st.write(f"{dataset_name}: Silhouette Score = {score:.2f}, {decision_here}")
                
            # Proceed with CNN model training for the best dataset
            best_dataset_file = next((x for x in uploaded_files if x.name == best_dataset_name), None)
            if best_dataset_file:
                images, labels, label_map = process_zip(best_dataset_file, target_size=(64, 64))
                X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
                
                model = create_cnn_model(input_shape=(64, 64, 3), num_classes=len(label_map))
                
                # Data augmentation
                train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
                val_datagen = ImageDataGenerator(rescale=1./255)
                
                history = model.fit(
                    train_datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=val_datagen.flow(X_val, y_val, batch_size=32),
                    epochs=10  # Adjust epochs as needed
                )
                
                # Save and download the model
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_file_path = os.path.join(temp_dir, "model.h5")
                    model.save(model_file_path)
                    
                    with open(model_file_path, 'rb') as model_file:
                        st.download_button(label="Download Trained Model", data=model_file, file_name="model.h5", mime="application/octet-stream")
    else:
        st.info("Please upload one or more ZIP files to proceed.")

if __name__ == '__main__':
    app3()
