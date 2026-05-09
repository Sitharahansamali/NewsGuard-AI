import streamlit as st
import requests

st.set_page_config(page_title="Fake News Detection AI")

st.title("Fake News Detection AI")

# TEXT INPUT

news = st.text_area("Enter News Text")

if st.button("Predict News"):

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        params={"text": news}
    )

    result = response.json()

    st.write(result)

    prediction = result["prediction"]

    st.success(prediction)

    st.info(
        f"Confidence: "
        f"{result['confidence']*100:.2f}%"
    )


st.divider()

# URL INPUT

url = st.text_input("Enter News URL")

if st.button("Analyze URL"):

    response = requests.post(
        "http://127.0.0.1:8000/predict_url",
        params={"url": url}
    )

    result = response.json()

    prediction = (
        "Real News"
        if result["prediction"] == 1
        else "Fake News"
    )
    st.subheader(result["title"])

    st.success(result["prediction"])

    st.info(
        f"Final Verification Score: "
        f"{result['final_score']}%")
    
    st.write(
        f"ML Confidence: "
        f"{result['ml_confidence']}%")
    
    st.write(
        f"Source: "
        f"{result['source']}")
    
    st.write(
        f"Credibility Score: "
        f"{result['credibility_score']}"
    )
    
    st.write(
        f"Credibility Level: "
        f"{result['credibility_level']}"
    )

