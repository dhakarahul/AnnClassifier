import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#Loading the train model
model = tf.keras.models.load_model('model.h5')

#loading one hot encoder for geo
with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)
#loading Label encoder for Gender Data
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
#Loading standard scler trained model for scaling the input values
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


#streamlit

st.title('Predicting customer churn')

#taking the input from the user

geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 70, 2)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 5,30, 2)
num_of_products = st.slider('No of Products', 1,4,1)
has_cr_card = st.selectbox('Has Credit Card?', [1,0])
is_active_member = st.selectbox('Is Active Member?', [1,0])

#preparing the input data 

input_data = {
    'Credit_Score': credit_score,
    'Gender': gender,
    'Age': age,
    'Balance': balance,
    'Credit Score': credit_score,
    'Estimated Salary': estimated_salary,
    'Tenure' : tenure,
    'No of Products': num_of_products,
    'Has Credit Card?' : has_cr_card,
    'Is Active Member?': is_active_member
}


#input_dataframe

# Creating input dataframe with correct column names
input_data = pd.DataFrame({
    'CreditScore': [credit_score],  # Match exact expected column name
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Balance': [balance],
    'EstimatedSalary': [estimated_salary],  # Match expected column name
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],  # Match expected column name
    'HasCrCard': [has_cr_card],  # Match expected column name
    'IsActiveMember': [is_active_member],  # Match expected column name
})


#encoding for the geography value
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded, columns = one_hot_encoder_geo.get_feature_names_out(['Geography']))

#combining it with input data 
input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis = 1)

#scaling the input data

# Ensure column order matches the order used during training
expected_columns = scaler.feature_names_in_  # Extracting the original feature order
input_data = input_data[expected_columns]  # Reordering columns to make sure that the input matches for scaler


scaled_data = scaler.transform(input_data)

#predicting the churn

prediction = model.predict(scaled_data)
prediction_prob = np.round(prediction[0][0],3)
st.write(f'The probability of Churn is {prediction_prob}')

if prediction_prob > 0.5:
    st.write('Customer is going to churn')
else:
    st.write('Customer is not going to churn')
