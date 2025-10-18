import streamlit as st
import joblib
import pandas as pd

# Load the saved regularized Decision Tree model
model = joblib.load('regularized_decision_tree_model.pkl')

st.title('Heart Disease Prediction')

# Get input from the user for each feature
age = st.number_input('Age', min_value=0, max_value=120, value=55)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3], format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=50, max_value=250, value=130)
chol = st.number_input('Serum Cholesterol (chol)', min_value=50, max_value=600, value=220)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[0, 1], format_func=lambda x: 'True' if x == 1 else 'False')
restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', options=[0, 1, 2], format_func=lambda x: ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'][x])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=50, max_value=250, value=160)
exang = st.selectbox('Exercise Induced Angina (exang)', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
oldpeak = st.number_input('ST depression induced by exercise relative to rest (oldpeak)', min_value=0.0, max_value=7.0, value=1.0, step=0.1)
slope = st.selectbox('Slope of the peak exercise ST segment (slope)', options=[0, 1, 2], format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
ca = st.number_input('Number of major vessels (0-3) colored by fluoroscopy (ca)', min_value=0, max_value=4, value=0)
thal = st.selectbox('Thalassemia (thal)', options=[0, 1, 2, 3], format_func=lambda x: ['Unknown', 'Normal', 'Fixed defect', 'Reversible defect'][x])

user_input = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

# Convert the dictionary to a pandas DataFrame
# Ensure the order of columns matches the training data
input_df = pd.DataFrame([user_input], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                 'ca', 'thal'])
prediction = model.predict(input_df)

# Display the prediction
st.subheader('Prediction Result')
if prediction[0] == 1:
    st.write('Based on the provided information, the model predicts: **Heart Disease Present**')
else:
    st.write('Based on the provided information, the model predicts: **No Heart Disease Present**')
