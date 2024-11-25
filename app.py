# Streamlit app for CO2 Emission Mitigation Predictor
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler, MultiLabelBinarizer

# Load pre-trained model and preprocessor
@st.cache_data
def load_model():
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# File uploader
st.title("CO2 Emission Mitigation Predictor")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Preprocessing functions
def create_dummy_variables_with_mlb(df, column_name):
    df[column_name] = df[column_name].apply(eval)
    mlb = MultiLabelBinarizer()
    binarized_data = mlb.fit_transform(df[column_name])
    binarized_df = pd.DataFrame(binarized_data, columns=mlb.classes_)
    df = pd.concat([df, binarized_df], axis=1).drop(columns=column_name)
    return df

def preprocess_data(df):
    df.columns = df.columns.str.replace(" ", "_")
    df = create_dummy_variables_with_mlb(df, 'Recycling')
    df = create_dummy_variables_with_mlb(df, 'Cooking_With')
    df['Gender'] = df['Gender'].map({'male': True, 'female': False})
    return df

# Prediction and results
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset Preview:", df.head())
    
    df_processed = preprocess_data(df)
    preprocessor = ColumnTransformer([
        ("numerical", MinMaxScaler(), [
            'Monthly_Grocery_Bill', 'Vehicle_Monthly_Distance_Km',
            'Waste_Bag_Weekly_Count', 'How_Long_TV_PC_Daily_Hour',
            'How_Many_New_Clothes_Monthly', 'How_Long_Internet_Daily_Hour'
        ]),
        ("categorical", OneHotEncoder(), [
            'Body_Type', 'Diet', 'How_Often_Shower', 'Social_Activity',
            'Frequency_of_Traveling_by_Air', 'Waste_Bag_Size', 'Energy_efficiency'
        ]),
    ], remainder="passthrough")
    
    X_transformed = preprocessor.fit_transform(df_processed)
    predictions = model.predict(X_transformed).round(2)
    df['Predicted_Carbon_Emission'] = predictions
    
    st.write("Dataset with Predictions:", df.head())
    
    # Download predictions
    @st.cache_data
    def convert_df_to_csv(dataframe):
        return dataframe.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(df)
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
