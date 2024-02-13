import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

def perform_pca(data):
    # Handle categorical variables with label encoding
    label_encoders = {}
    for i in range(data.shape[1]):
        if isinstance(data[0, i], str):
            label_encoders[i] = LabelEncoder()
            data[:, i] = label_encoders[i].fit_transform(data[:, i])

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA()
    pca.fit(data_scaled)
    data_reduced = pca.transform(data_scaled)

    return data_reduced, pca.explained_variance_ratio_, pca.components_

def main():
    st.title("PCA Visualization")

    # Upload dataset
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Display the dataset
        st.subheader("Original Dataset")
        st.write(data)

        # Perform PCA
        data_array = data.values
        data_reduced, explained_variance_ratio, components = perform_pca(data_array)

        # Display explained variance ratio
        st.subheader("Explained Variance Ratio")
        st.write(explained_variance_ratio)

        # Display the reduced dataset
        st.subheader("Reduced Dataset")
        reduced_df = pd.DataFrame(data_reduced, columns=[f"PC{i+1}" for i in range(data_reduced.shape[1])])
        st.write(reduced_df)

        # Display top features for each principal component
        st.subheader("Top Features for Each Principal Component")
        for i, component in enumerate(components):
            st.write(f"Principal Component {i+1}")
            top_features_idx = component.argsort()[-3:][::-1]
            top_features_names = data.columns[top_features_idx]
            st.write(top_features_names)

if __name__ == "__main__":
    main()
