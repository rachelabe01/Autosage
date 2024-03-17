import streamlit as st
import pickle

def load_model(model_path):
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def predict_medv(model, input_values):
    predicted_output = model.predict(input_values)
    return predicted_output

def main():
    st.title("MEDV Prediction App")

    # Load the pickled model
    model_path = r'C:/Users/rachel/OneDrive/Desktop/trying/housing (7)_model.pkl'
    loaded_model = load_model(model_path)

    # Provide input values for features RM, LSTAT, and PTRATIO
    input_values = st.text_input("Enter input values for RM, LSTAT, and PTRATIO (comma-separated):")
    if input_values:
        input_values = list(map(float, input_values.split(',')))
        input_values = [input_values]  # Convert to 2D array

        # Use the loaded model to predict MEDV values for the provided input
        predicted_output = predict_medv(loaded_model, input_values)

        # Print the predicted MEDV value
        st.write("Predicted MEDV value:", predicted_output[0])

if __name__ == "__main__":
    main()
