import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import precision_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import pickle
import base64

def identify_problem_type(df):
    target_column = df.iloc[:, -1]
    unique_values = target_column.unique()
    num_unique_values = len(unique_values)
    if num_unique_values < 3:
        return 'Classification'
    else:
        # Use PCA to analyze the target column
        pca = PCA(n_components=1)
        target_pca = pca.fit_transform(target_column.values.reshape(-1, 1))
        if pca.explained_variance_ratio_[0] > 0.95:
            return 'Regression'
        else:
            return 'Classification'

def label_encode_categorical(df):
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    return df

def evaluate_classification(df):
    X = df.drop(columns=[df.columns[-1]])
    y = df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return precision, f1, model

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
def train_and_evaluate_classification(df, model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    st.write(f"Model: {model_name}")
    st.write(f"Precision: {precision}")
    st.write(f"F1 Score: {f1}")

    # Save the trained model using pickle
    with open(f"{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

    # Display download link
    st.markdown(f"[Download {model_name}.pkl](/download/{model_name}.pkl)", unsafe_allow_html=True)

def train_and_evaluate_regression(df, model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Model: {model_name}")
    
    st.write(f"R2 Score: {r2}")

    # Save the trained model using pickle
    with open(f"{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

    # Display download link
    st.markdown(f"[Download {model_name}.pkl](/download/{model_name}.pkl)", unsafe_allow_html=True)


def compare_datasets(dataset_info):
    sorted_datasets = sorted(dataset_info, key=lambda x: (x['f1_score'] if x['problem_type'] == 'Classification' else -x['mse']), reverse=True)
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

            df = label_encode_categorical(df)

            if problem_type == 'Classification':
                precision, f1, model = evaluate_classification(df)
                metrics = {'precision_score': precision, 'f1_score': f1}
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
                st.write(f"   F1 Score: {dataset['f1_score']}, Precision Score: {dataset['precision_score']}")
            elif dataset['problem_type'] == 'Regression':
                st.write(f"   Mean Squared Error: {dataset['mse']}, R2 Score: {dataset['r2_score']}")

        # Select top-ranked dataset
        top_dataset = top_datasets[0]
        st.subheader(f"Modeling Top Ranked Dataset: {top_dataset['name']}")
        st.write(f"Problem Type: {top_dataset['problem_type']}")

        # Display model accuracy
        if top_dataset['problem_type'] == 'Classification':
            st.write(f"Model F1 Score: {top_dataset['f1_score']}, Precision Score: {top_dataset['precision_score']}")
        elif top_dataset['problem_type'] == 'Regression':
            st.write(f"Model Mean Squared Error: {top_dataset['mse']}, R2 Score: {top_dataset['r2_score']}")

        # Save model
        # Save model
        st.write("Do you want to download the trained model?")
        model_name = top_dataset['name'].split('.')[0] + '_model.pkl'
        with open(model_name, 'wb') as f:
            pickle.dump(top_dataset['model'], f)
        # Read the saved model to display as a download button
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