import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    # else:
    #     return 'Classification'

# Function to display feature importance
def display_feature_importance(X, y, problem_type):
    model = RandomForestRegressor() if problem_type == 'Regression' else RandomForestClassifier()
    model.fit(X, y)

    feature_importances = model.feature_importances_

    return pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)

# Function to find potential target columns using PCA
def find_potential_target_columns(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    num_components = np.argmax(cumulative_variance_ratio > 0.95) + 1
    
    # Get component names
    component_names = [f"Component {i+1}" for i in range(num_components)]
    
    # Get explained variance ratio
    explained_variance = pca.explained_variance_
    
    # Get feature names associated with each component
    feature_names = list(X.columns)
    components_features = []
    for i in range(num_components):
        component_feature_weights = pca.components_[i]
        component_feature_names = [feature_names[j] for j in np.argsort(np.abs(component_feature_weights))[::-1]]
        components_features.append(component_feature_names)
    
    return pd.DataFrame({'Component Name': component_names, 
                         'Explained Variance Ratio': explained_variance_ratio,
                         'Component Features': components_features})

def main():
    st.title('Machine Learning Problem Type Detection and Analysis')

    # Upload file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
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

        # Feature Importance
        if problem_type == 'Regression':
            X = dataset.iloc[:, :-1]
            y = target_column
            feature_importances = display_feature_importance(X, y, problem_type)
            st.subheader('Important Features:')
            st.write(feature_importances.head(5))  # Display top 5 important features for regression
        else:
            X = dataset.iloc[:, :-1]
            component_info = find_potential_target_columns(X)
            st.subheader('PCA Important Features:')
            st.write(component_info)

if __name__ == '__main__':
    main()