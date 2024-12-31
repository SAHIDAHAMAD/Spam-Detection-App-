import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import base64

# Load trained model and vectorizer
with open('spam_model.pkl', 'rb') as file:
    model_data = pickle.load(file)
model = model_data['model']
cv = model_data['vectorizer']
ps = PorterStemmer()

# Text preprocessing function
def preprocess_text(text):
    rp = re.sub('[^a-zA-Z]', " ", text)
    rp = rp.lower()
    rp = rp.split()
    rp = [ps.stem(word) for word in rp if word not in set(stopwords.words('english'))]
    return " ".join(rp)

# Function to set a background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
        }}
        .title {{
            font-size: 48px;
            font-weight: bold;
            text-align: center;
        }}
        .input-label {{
            font-size: 24px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
set_background("email.jpg")  # Replace with your image file name

# App title
st.markdown("<div class='title'>📧 Spam Detection App</div>", unsafe_allow_html=True)

# Input section
st.markdown("<div class='input-label'>Enter the email text below:</div>", unsafe_allow_html=True)
user_input = st.text_area("", height=200)

if st.button("Classify"):
    if user_input:
        # Preprocess and transform input
        processed_text = preprocess_text(user_input)
        input_vector = cv.transform([processed_text]).toarray()
        
        # Predict
        prediction = model.predict(input_vector)[0]
        label = "Spam" if prediction == 1 else "Not Spam"
        
        # Display result with colors
        if label == "Not Spam":
            st.success(f"The email is classified as **{label}**.")
        else:
            st.error(f"The email is classified as **{label}**.")
    else:
        st.warning("Please enter some text to classify.")

# Additional Features
st.markdown("---")
st.markdown("#### Additional Features:")
uploaded_image = st.file_uploader("Upload a Background Image (Optional):", type=["jpg", "jpeg", "png"])
if uploaded_image:
    set_background(uploaded_image.name)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Developed by <b>Sahid Ahamad</b></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>Powered by Machine Learning with Streamlit</div>", unsafe_allow_html=True)

