# app1.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, r2_score, silhouette_score
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import pickle
import base64

def app1():
    def find_potential_target_column(df):
        df_numeric = df.select_dtypes(include=['number'])  # Select only numeric columns
        df_numeric.dropna(axis=1, how='all', inplace=True)  # Drop columns with all NaN values

        if len(df_numeric.columns) > 2000:
            return find_potential_target_column_using_silhouette_score(df_numeric)
        else:
            return find_potential_target_column_using_correlation(df_numeric)

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

    def evaluate_top_dataset(df_scaled):
        X = df_scaled.drop(columns=[df_scaled.columns[-1]])
        y = df_scaled[df_scaled.columns[-1]]
        
        if identify_problem_type(df_scaled) == 'Classification':
            # Convert target variable to categorical for classification
            y = y.astype(int).astype(str)

            models = {
                'Random Forest Classifier': RandomForestClassifier(),
                'Hist Gradient Boosting Classifier': HistGradientBoostingClassifier(),
                'Logistic Regression': LogisticRegression()
            }
        else:
            models = {
                'Random Forest Regressor': RandomForestRegressor(),
                'Hist Gradient Boosting Regressor': HistGradientBoostingRegressor(),
                'Linear Regression': LinearRegression()
            }
        
        results = []
        for model_name, model in models.items():
            model.fit(X, y)
            y_pred = model.predict(X)
            
            if identify_problem_type(df_scaled) == 'Classification':
                precision = precision_score(y, y_pred, average='weighted')
                results.append({'Model': model_name, 'Precision': precision})
                st.write(f"{model_name} Precision: {precision}")
            else:
                r2 = r2_score(y, y_pred)
                results.append({'Model': model_name, 'R2 Score': r2})
                st.write(f"{model_name} R2 Score: {r2}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find the best performing model
        if identify_problem_type(df_scaled) == 'Classification':
            best_model_name = results_df.loc[results_df['Precision'].idxmax()]['Model']
            best_model_precision_or_r2 = results_df.loc[results_df['Precision'].idxmax()]['Precision']
            best_model = models[best_model_name]
        else:
            best_model_name = results_df.loc[results_df['R2 Score'].idxmax()]['Model']
            best_model_precision_or_r2 = results_df.loc[results_df['R2 Score'].idxmax()]['R2 Score']
            best_model = models[best_model_name]

        # Display the best model and its performance
        st.write(f"Best Model: {best_model_name}")
        if identify_problem_type(df_scaled) == 'Classification':
            st.write(f"Best Model Precision: {best_model_precision_or_r2}")
        else:
            st.write(f"Best Model R2 Score: {best_model_precision_or_r2}")

        # Serialize and encode the best model for download
        serialized_best_model = pickle.dumps(best_model)
        b64_encoded_model = base64.b64encode(serialized_best_model).decode()
        href = f'<a href="data:file/txt;base64,{b64_encoded_model}" download="{best_model_name}.pkl">Download {best_model_name}</a>'
        st.markdown(href, unsafe_allow_html=True)

    st.title("Automated Textbased Model")

    # Upload dataset
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Read dataset
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        
        # Preprocess dataset
        df_scaled = preprocess_data(df)
        
        # Evaluate dataset and provide download link for the best model
        evaluate_top_dataset(df_scaled)

# If this script is run directly (instead of being imported), just display the app's functionality
if __name__ == '__main__':
    app1()
