import streamlit as st
import numpy as np
import pandas as pd
import requests

st.title("Simple command web app")

uploaded_file = st.sidebar.file_uploader(
    "Upload audio file of command", accept_multiple_files=False)

if uploaded_file is not None:
    st.write(uploaded_file.name)
    st.write(uploaded_file)
    st.audio(uploaded_file)
    if st.button("Get the prediction for this audio"):
        
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        prediction = requests.post(
            "http://localhost:3000/classify_file",
            data=bytes_data,
            headers={'content-type': 'application/octet-stream'}
        )
        st.write(prediction.text)
else:
    st.write("Try uploading a file!")
