import streamlit as st
import pandas as pd
import requests


# Specify the URL name 
API_URL = r"http://localhost:8000/predict"       # Give your external url to host the model

# Set title of the page
st.title("Consumer activity predictor app")

# Set markdown
st.markdown("Enter your details below")

# set the value of the required field
Agency = st.text_input("Name of the Agency that shows the products", value = "City Charter")
month_lst = [i for i in range(1, 13)]
Month_Sampled = st.selectbox("On which Month the product has been shown", options = month_lst)
fb_data = st.number_input("Likes on the product at the facebbok platform", min_value = 1.0, value = 291.90)


# Create a button
if st.button("Predict consumer activity"):
    input_data = {
        "Agency" : Agency,
        "Month_Sampled" : Month_Sampled,
        "fb_data" : fb_data
    }

    try:
        response = requests.post(API_URL, json = input_data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted consumer activity on twitter : **{result['twitter']}**")
        else:
            st.error(f"API error : {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to fastapi server. Make sure it's running on 8000 port")
