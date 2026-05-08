import streamlit as st
import requests

# Page Title
st.set_page_config(page_title="Fake News Detection AI")

st.title("Fake News Detection AI")
st.write("Enter a news article or headline to check whether it is Fake or Real.")

# Text Input
news = st.text_area("Enter News Text")

# Predict Button
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                 json={"text": news}
            )

            result = response.json()

            prediction = result["prediction"]

            if prediction == "Fake News":
                st.error("This news is predicted as FAKE.")
            else:
                st.success("This news is predicted as REAL.")

            st.write(result)

        except:
            st.error("Could not connect to FastAPI server.")