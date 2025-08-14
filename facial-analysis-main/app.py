import streamlit as st
import requests
import base64

# Function to set background image on the left
def set_custom_css(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-color: #121212;
        overflow: hidden;
        height: 100vh;
        display: flex;
        flex-direction: row;
    }}

    .left-container {{
        width: 40vw;
        height: 100vh;
        background-image: url('data:image/png;base64,{encoded_string}');
        background-size: cover;
        background-position: center;
        position: fixed;
        left: 0;
        top: 0;
    }}

    .right-container {{
        width: 60vw;
        height: 100vh;
        position: fixed;
        right: 0;
        top: 0;
        padding: 40px;
        padding-left: 150px;
        color: white;
        overflow-y: auto;
    }}

    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 10px;
        width: 100%;
        border: none;
    }}

    .stTextInput input, .stFileUploader {{
        background-color: #2A2A2A;
        color: white !important;
    }}

    .stMarkdown, label, .stTextInput label, .stFileUploader label, .stRadio label, .stRadio div {{
        color: white !important;
        font-size: 16px;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: white !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the CSS setup function
set_custom_css("assets/logo/logo.png")

# Left container placeholder
st.markdown("<div class='left-container'></div>", unsafe_allow_html=True)

# Right-side Functional UI
st.markdown("<div class='right-container'>", unsafe_allow_html=True)

st.title("Facial Recognition System")

# User Registration
st.subheader("Register a New User")
name = st.text_input("Enter Name")
image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="register")

if st.button("Register"):
    if name and image:
        files = {"image": image.getvalue()}
        data = {"name": name}
        try:
            response = requests.post("http://127.0.0.1:5000/register", files=files, data=data)
            if response.status_code == 200:
                st.success(f"User {name} registered successfully!")
            else:
                st.error("Registration failed. Please try again.")
        except requests.exceptions.RequestException:
            st.error("Connection to server failed. Please try again later.")
    else:
        st.warning("Please enter a name and upload an image.")

# Face Recognition
st.subheader("Face Recognition")
option = st.radio("Choose Input Method:", ("Live Camera", "Upload Image"))

if option == "Live Camera":
    if st.button("Start Camera Prediction"):
        try:
            response = requests.get("http://127.0.0.1:5000/predict")
            if response.status_code == 200:
                st.success("Camera prediction started!")
            else:
                st.error("Failed to start prediction. Please try again.")
        except requests.exceptions.RequestException:
            st.error("Connection to server failed. Please try again later.")

elif option == "Upload Image":
    uploaded_image = st.file_uploader("Upload Image for Prediction", type=["jpg", "jpeg", "png"], key="predict")
    if uploaded_image:
        if st.button("Predict Uploaded Image"):
            files = {"image": uploaded_image.getvalue()}
            try:
                response = requests.post("http://127.0.0.1:5000/predict_image", files=files)
                if response.status_code == 200:
                    prediction = response.json().get("prediction", {})
                    st.success("Prediction Completed:")
                    st.json(prediction)
                else:
                    st.error("Prediction failed. Please try again.")
            except requests.exceptions.RequestException:
                st.error("Connection to server failed. Please try again later.")

st.markdown("</div>", unsafe_allow_html=True)