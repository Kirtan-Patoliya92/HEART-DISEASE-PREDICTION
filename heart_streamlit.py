import streamlit as st
import joblib as jb
import pandas as pd

model = jb.load('model_heart.pkl')
columns = jb.load('model_columns.pkl')
scaler = jb.load('model_heart_scaler.pkl')


st.title("Heart Disease Prediction ML Model ❤️")


Age = st.number_input(
    "Enter your age",
    min_value=18,
    max_value=99
)


RestingBP = st.number_input(
    "Enter your resting BP score".upper(),
    min_value=80,
    max_value=200
)


Cholesterol = st.number_input(
    "Enter your cholesterol score".upper(),
    min_value=60,
    max_value=600
)


FastingBS = st.selectbox(
    "Are you struggling with fasting blood sugar?".upper(),
    ["No", "Yes"]
)

if FastingBS == "No":
    FastingBS = 0
else:
    FastingBS = 1


MaxHR = st.number_input(
    "Enter your MaxHR score".upper(),
    min_value=60,
    max_value=203
)


Oldpeak = st.number_input(
    "Enter your Oldpeak score".upper(),
    min_value=-5.0,
    max_value=7.0
)


Sex_M = st.selectbox(
    "Enter your gender".upper(),
    ["Female", "Male"]
)

if Sex_M == "Female":
    Sex_M = 0
else:
    Sex_M = 1


Chestpain_type = st.selectbox(
    "Select your chest pain type".upper(),
    [
        "ChestPainType_ATA",
        "ChestPainType_NAP",
        "ChestPainType_TA"
    ]
)


if Chestpain_type == "ChestPainType_ATA":
    ChestPainType_ATA = 1
    ChestPainType_NAP = 0
    ChestPainType_TA = 0

elif Chestpain_type == "ChestPainType_NAP":
    ChestPainType_ATA = 0
    ChestPainType_NAP = 1
    ChestPainType_TA = 0

else:
    ChestPainType_ATA = 0
    ChestPainType_NAP = 0
    ChestPainType_TA = 1


resting_egc = st.selectbox(
    "Select your resting ECG type".upper(),
    ["Normal", "ST"]
)


if resting_egc == "Normal":
    RestingECG_Normal = 1
    RestingECG_ST = 0
else:
    RestingECG_Normal = 0
    RestingECG_ST = 1


ExerciseAngina_Y = st.selectbox(
    "Select exercise angina",
    ["NO", "YES"]
)


if ExerciseAngina_Y == "NO":
    ExerciseAngina_Y = 0
else:
    ExerciseAngina_Y = 1


st_slop = st.selectbox(
    "Select your ST slope".upper(),
    ["Flat", "Up"]
)


if st_slop == "Flat":
    ST_Slope_Flat = 1
    ST_Slope_Up = 0
else:
    ST_Slope_Flat = 0
    ST_Slope_Up = 1


if st.button("Predict"):

    row_data = {
        "Age": Age,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "MaxHR": MaxHR,
        "Oldpeak": Oldpeak,
        "Sex_M": Sex_M,

        "ChestPainType_ATA": ChestPainType_ATA,
        "ChestPainType_NAP": ChestPainType_NAP,
        "ChestPainType_TA": ChestPainType_TA,

        "RestingECG_ST": RestingECG_ST,
        "RestingECG_Normal": RestingECG_Normal,

        "ExerciseAngina_Y": ExerciseAngina_Y,

        "ST_Slope_Flat": ST_Slope_Flat,
        "ST_Slope_Up": ST_Slope_Up
    }


    final_data = pd.DataFrame([row_data])


    # Remove target column
    input_columns = [
        col for col in columns
        if col != "HeartDisease"
    ]


    # Match training column order
    final_data = final_data.reindex(
        columns=input_columns,
        fill_value=0
    )


    final_data = final_data.astype(float)


    # Scale input
    final_data = scaler.transform(final_data)


    # Prediction
    prediction = model.predict(final_data)

    prediction = prediction[0]


    if prediction == 0:
        st.success(
            "Prediction: No heart disease detected."
        )

    elif prediction == 1:
        st.error(
            "Prediction: Heart disease detected."
        )