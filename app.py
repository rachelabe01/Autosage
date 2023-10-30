import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def compare_datasets(dataset_info):
    sorted_datasets = sorted(dataset_info, key=lambda x: (x['f1_score'], x['precision_score']), reverse=True)
    return sorted_datasets[:3]


def evaluate_dataset(df):
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return precision, f1


def main():
    st.title("Dataset Comparison App")

    num_datasets = st.number_input("Enter the number of datasets:", min_value=1, value=1, step=1)

    dataset_info = []
    for i in range(1, num_datasets + 1):
        dataset_name = st.text_input(f"Enter the name of dataset {i}:")
        dataset_file = st.file_uploader(f"Upload CSV file for dataset {i}:", type=["csv"])

        if dataset_file is not None:
            df = pd.read_csv(dataset_file)
            num_rows = df.shape[0]
            num_nulls = df.isnull().sum().sum()
            precision, f1 = evaluate_dataset(df)
            dataset_info.append({'name': dataset_name, 'num_rows': num_rows, 'num_nulls': num_nulls,
                                 'precision_score': precision, 'f1_score': f1})



    if st.button("Compare"):
        if num_datasets < 2:
            st.error("Please enter a valid number of datasets (minimum is 2).")
            return  # Exit the function if num_datasets is less than 2
        elif len(dataset_info) != num_datasets:
            st.error("Please upload the specified number of datasets.")
            return  # Exit the function if the number of uploaded datasets does not match num_datasets
        top_datasets = compare_datasets(dataset_info)
        if num_datasets <= 3:
            st.success("The top {} datasets for your project are:".format(num_datasets))
        else:
            st.success("The top  datasets for your project are:")
        for idx, dataset in enumerate(top_datasets):
            st.write(
                f"{idx + 1}. Name: {dataset['name']}, F1 Score: {dataset['f1_score']}, Precision Score: {dataset['precision_score']}")

        # Create a bar graph
        dataset_names = [dataset['name'] for dataset in top_datasets]
        f1_scores = [dataset['f1_score'] for dataset in top_datasets]
        precision_scores = [dataset['precision_score'] for dataset in top_datasets]

        plt.figure(figsize=(10, 6))
        plt.bar(dataset_names, f1_scores, alpha=0.7, label='F1 Score', width=0.4)
        plt.bar(dataset_names, precision_scores, alpha=0.7, label='Precision', width=0.4, align='edge')
        plt.xlabel('Datasets')
        plt.ylabel('Scores')
        plt.title('Comparison of Datasets')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()
