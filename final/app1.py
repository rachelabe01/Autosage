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
        
        return best_model_name, best_model_precision_or_r2, results_df, best_model  # Return the necessary values

    def get_download_link(model_bytes, model_name):
        b64_encoded_model = base64.b64encode(model_bytes).decode()
        href = f'<a href="data:file/txt;base64,{b64_encoded_model}" download="{model_name}.pkl">Download {model_name}</a>'
        return href

    st.title("Automated Textbased Model")

    # Upload dataset
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

    dataset_info = []
    dataframes = {}  # Dictionary to store DataFrames in memory
    df_comparison = None  # Define df_comparison variable
    
    if uploaded_files:
        for file in uploaded_files:
            file_name = file.name
            df = pd.read_csv(file)
            # Store the DataFrame in memory
            dataframes[file_name] = df
            st.write(f"**{file_name}**")
            st.write(df.head())

            df_scaled = preprocess_data(df)

            problem_type = identify_problem_type(df_scaled)
            st.write(f"Problem Type: {problem_type}")

            # Compute correlation with t-SNE components
            potential_target_column = find_potential_target_column(df)
            correlation = np.corrcoef(df[potential_target_column], df_scaled.iloc[:, :-1], rowvar=False)[0, 1:]

            dataset_info.append({'name': file_name, 'problem_type': problem_type, 'correlation_with_tsne': correlation, 'model': None})

    if dataset_info:
        st.subheader("Dataset Comparison")
        df_comparison = pd.DataFrame(dataset_info)
        df_comparison['correlation_with_tsne'] = df_comparison['correlation_with_tsne'].apply(lambda x: np.abs(x).mean())  # Use mean correlation
        df_comparison.sort_values(by='correlation_with_tsne', ascending=False, inplace=True)
        st.write(df_comparison)

        if st.button("Download Top Ranked Model"):
            top_dataset_name = df_comparison.iloc[0]['name']
            # Retrieve the DataFrame from memory instead of reading from a file
            df = dataframes[top_dataset_name]

            df_scaled = preprocess_data(df)
            best_model_name, best_model_precision_or_r2, results_df, best_model = evaluate_top_dataset(df_scaled)

            # Save model to a binary stream instead of a file to facilitate download
            model_name = top_dataset_name.split('.')[0] + '_model.pkl'
            model_bytes = pickle.dumps(best_model)

            # Print the information about the best model
            st.write(f"The best model for {top_dataset_name} is: {best_model_name}")
            st.write(f"Performance Metric: {best_model_precision_or_r2}")

            # Display download link
            st.write("Do you want to download the trained model?")
            st.markdown(get_download_link(model_bytes, model_name), unsafe_allow_html=True)
            
            # Display performance metrics in tabular format
            st.subheader("Performance Metrics")
            st.table(results_df)

# If this script is run directly (instead of being imported), just display the app's functionality
if __name__ == '__main__':
    app1()
