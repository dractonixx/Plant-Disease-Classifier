import streamlit as st
import pandas as pd
import random
import streamlit.components.v1 as components
from recognise import predict
import tempfile
import os

st.set_page_config(
    page_title="P L A N T",
    page_icon=":dna:",
    initial_sidebar_state="expanded",
)

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

components.html(
    """
    <style>
        #effect{
            margin:0px;
            padding:0px;
            font-family: "Source Sans Pro", sans-serif;
            font-size: max(8vw, 20px);
            font-weight: 700;
            top: 0px;
            right: 25%;
            position: fixed;
            background: -webkit-linear-gradient(0.25turn,#FF4C4B, #FFFB80);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p{
            font-size: 2rem;
        }
    </style>
    <p id="effect">P L A N T</p>
    """,
    height=69,
)

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Make prediction (get only top 1)
        with st.spinner('Analyzing plant disease...'):
            results = predict(tmp_path, top_k=1)
        
        # Display result (only one prediction)
        disease, confidence = results[0]
        
        st.success("Analysis Complete!")
        st.subheader(f"Detected: {disease}")
        st.progress(confidence)
        st.write(f"**Confidence: {confidence*100:.2f}%**")
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)