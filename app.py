import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle


# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model("model.h5")


# ==========================================
# LOAD ENCODERS AND SCALER
# ==========================================

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("Scaler.pkl", "rb") as file:
    Scaler = pickle.load(file)


# ==========================================
# STREAMLIT UI
# ==========================================

st.title("CUSTOMER CHURN PREDICTION")

st.write("Enter customer details below:")


# Geography
geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)


# Gender
gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)


# Age
age = st.slider(
    "Age",
    min_value=18,
    max_value=92,
    value=30
)


# Credit Score
credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)


# Balance
balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)


# Estimated Salary
estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=100000.0
)


# Tenure
tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)


# Number of Products
num_of_products = st.slider(
    "Number Of Products",
    min_value=1,
    max_value=4,
    value=1
)


# Credit Card
has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)


# Active Member
is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)


# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("Predict Churn"):

    # ======================================
    # CREATE INPUT DATAFRAME
    # ======================================

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [
            label_encoder_gender.transform([gender])[0]
        ],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })


    # ======================================
    # ONE-HOT ENCODE GEOGRAPHY
    # ======================================

    geo_encoded = onehot_encoder_geo.transform(
        input_data[["Geography"]]
    )


    geo_encoded_df = pd.DataFrame(
        geo_encoded.toarray(),
        columns=onehot_encoder_geo.get_feature_names_out(
            ["Geography"]
        )
    )


    # ======================================
    # COMBINE DATA
    # ======================================

    input_df = pd.concat(
        [
            input_data.drop("Geography", axis=1),
            geo_encoded_df
        ],
        axis=1
    )


    # ======================================
    # CHECK COLUMN ORDER
    # ======================================

    # This is extremely important.
    # Scaler must receive the same columns
    # that were used during training.

    if hasattr(Scaler, "feature_names_in_"):

        input_df = input_df[
            Scaler.feature_names_in_
        ]


    # ======================================
    # SCALE DATA
    # ======================================

    input_scaled = Scaler.transform(input_df)


    # ======================================
    # MODEL PREDICTION
    # ======================================

    prediction = model.predict(
        input_scaled,
        verbose=0
    )


    prediction_proba = float(
        prediction[0][0]
    )


    # ======================================
    # DISPLAY PROBABILITY
    # ======================================

    st.subheader("Prediction Result")

    st.write(
        f"Churn Probability: {prediction_proba:.2%}"
    )


    # ======================================
    # DISPLAY RESULT
    # ======================================

    if prediction_proba > 0.5:

        st.error(
            "THE CUSTOMER IS LIKELY TO CHURN."
        )

    else:

        st.success(
            "THE CUSTOMER IS NOT LIKELY TO CHURN."
        )