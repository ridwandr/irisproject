import streamlit as st
import pandas as pd
import numpy as np
import joblib

# set page config
st.set_page_config(
    page_title="iris classification",
    page_icon=":cherry_blossom:",
    layout="centered",
)

# load model
model = joblib.load('iris_model.joblib')

# get prediction function
def get_prediction(data:pd.DataFrame, model):
    """Get Prediction
    
    args:
        data (pd.DataFrame):dataframe
        model (_type_): model classifier
    
    """
    prediction = model.predict(data)
    prediction_prob = model.predict_proba(data)

    map_label = {0:"setosa", 1:"versicolor", 2:"virginica"}
    prediction_label = map(lambda x: map_label[x], list(prediction))

    return {
        "prediction": prediction,
        "prob": prediction_prob,
        "label": list(prediction_label)
    }

st.title("iris classification", width="stretch")  # ngasih judul
st.write("get your iris species") # fungsi seperti print

sepal_info, petal_info= st.columns(2, gap="medium", border=True)

# sepal input
sepal_info.subheader("sepal information")
sepal_length = sepal_info.number_input("sepal length", min_value=0.0, max_value=10.0, value=0.0)
sepal_width = sepal_info.number_input("sepal width", min_value=0.0, max_value=10.0, value=0.0)

# petal input
petal_info.subheader("petal information")
petal_length = petal_info.number_input("petal length", min_value=0.0, max_value=10.0, value=0.0)
petal_width = petal_info.number_input("petal width", min_value=0.0, max_value=10.0, value=0.0)


predict = st.button("predict", use_container_width=True)

if predict:  # jika menekan button predict maka akan menampilkan input dalam bentuk dataframe
    df = pd.DataFrame(
        np.array([[sepal_length, sepal_width, petal_length, petal_width]]),
        columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
    )
    st.write(df)

    # predict
    result = get_prediction(df, model)
    label = result["label"][0]
    prediction = result["prediction"][0]
    prob = result["prob"][0][prediction]
    
    st.write(f"Your Iris Species is: **{prob:.0%} {label}**")
