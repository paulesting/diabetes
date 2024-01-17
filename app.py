import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

# Load the trained AdaBoostClassifier model
with open('original_classifier_AdaBoostClassifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Decision tree conditions and interpretations
decision_tree_conditions = {
    'N1': lambda row: (row['Glucose'] <= 120) & (row['Age'] <= 30),
    'N2': lambda row: row['BMI'] <= 30,
    'N3': lambda row: (row['Age'] <= 30) & (row['Pregnancies'] <= 6),
    'N4': lambda row: (row['Glucose'] <= 105) & (row['BloodPressure'] <= 80),
    'N5': lambda row: row['SkinThickness'] <= 20,
    'N6': lambda row: (row['BMI'] < 30) & (row['SkinThickness'] <= 20),
    'N7': lambda row: (row['Glucose'] <= 105) & (row['BMI'] <= 30),
    'N9': lambda row: row['Insulin'] < 200,
    'N10': lambda row: row['BloodPressure'] < 80,
    'N11': lambda row: (row['Pregnancies'] > 0) & (row['Pregnancies'] < 4),
    'N0': lambda row: row['BMI'] * row['SkinThickness'],
    'N8': lambda row: row['Pregnancies'] / row['Age'],
    'N13': lambda row: row['Glucose'] / row['DiabetesPedigreeFunction'],
    'N12': lambda row: row['Age'] * row['DiabetesPedigreeFunction'],
    'N14': lambda row: row['Age'] / row['Insulin'],
}

# Define the feature columns
input_feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Streamlit app
st.title('Diabetes Risk Prediction')

# Input form for user input
st.sidebar.header('User Input Features')
user_input = {}
for feature in input_feature_columns:
    user_input[feature] = st.sidebar.number_input(f"Enter {feature}", value=1.0, step=1.0)

predict_button = st.sidebar.button('Predict')

if predict_button:
    user_input_df = pd.DataFrame([user_input])

    # Decision tree conditions and interpretations
    conditions_result = {}
    for node, condition_func in decision_tree_conditions.items():
        feature_name = f"{node}"
        conditions_result[feature_name] = condition_func(user_input_df)

    # Convert conditions to DataFrame
    conditions_df = pd.DataFrame([conditions_result])

    # Combine original features and calculated conditions
    input_features = pd.concat([user_input_df[input_feature_columns], conditions_df], axis=1)
    input_features['N15'] = np.where(pd.to_numeric(input_features['N0'], errors='coerce') < 1034, 1, 0)

    feature_order = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N9', 'N10', 'N11', 'N15',
                     'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age', 'N0', 'N8', 'N13', 'N12', 'N14']

    # Reorder the columns
    input_features = input_features[feature_order]
    input_features = input_features.astype(float)
    st.sidebar.subheader('User Input:')
    st.sidebar.write(user_input_df)

    # Display calculated features
    st.sidebar.subheader('Calculated Features:')
    st.sidebar.write(input_features[['N0', 'N8', 'N13', 'N12', 'N14']])

    # Make predictions using the model
    prediction = model.predict(input_features)[0]

    # Display prediction

    st.subheader('Verdict:')
    if prediction == 1:
        st.write("<p style='color: white; font-size: 50px;'>The model predicts that the individual is at risk of diabetes.", unsafe_allow_html=True)
        st.markdown("<p style='color: red; font-size: 50px;'>Risk of Diabetes</p>", unsafe_allow_html=True)
    else:
        st.write("<p style='color: white; font-size: 50px;'>The model predicts that the individual is not at risk of diabetes.", unsafe_allow_html=True)
        st.markdown("<p style='color: green; font-size: 50px;'>No Risk of Diabetes</p>", unsafe_allow_html=True)


