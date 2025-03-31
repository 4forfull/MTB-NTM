import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="MTB and NTM Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model and image loading for better performance
@st.cache_resource
def load_model():
    try:
        model = joblib.load('./0.75-10_RF_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        return None

@st.cache_resource
def load_image():
    try:
        return Image.open('img.jpg')
    except FileNotFoundError:
        st.warning("Image file not found. Continuing without image.")
        return None

# Load resources
model = load_model()
image = load_image()

# Prediction function
def predict_diagnosis(input_data):
    try:
        test = np.array([input_data])
        prediction = model.predict(test)
        probability = model.predict_proba(test)[0]
        diagnosis = "MTB" if prediction[0] == 0 else "NTM"
        confidence = max(probability) * 100
        return diagnosis, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Main application
def main():
    # Header section
    if image:
        st.image(image, use_column_width=True)
    st.title("MTB and NTM Diagnosis Prediction")
    st.markdown("Enter patient parameters below to predict MTB/NTM diagnosis")

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    # Input fields with validation
    with col1:
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        alb = st.number_input("Albumin (35–50 g/L)", min_value=0.0, max_value=100.0, step=0.1)
        cl = st.number_input("Chloride (98–106 mmol/L)", min_value=0.0, max_value=200.0, step=0.1)
        cr = st.number_input("Creatinine (μmol/L)",
                            min_value=0.0,
                            max_value=200.0,
                            step=0.1,
                            help="Male: 59–104 μmol/L, Female: 45–84 μmol/L")
        glb = st.number_input("Globulin (20-30 g/L)", min_value=0.0, max_value=100.0, step=0.1)

    with col2:
        hdl = st.number_input("HDL Cholesterol (1.04–1.55 mmol/L)", min_value=0.0, max_value=5.0, step=0.01)
        na = st.number_input("Sodium (135–145 mmol/L)", min_value=0.0, max_value=200.0, step=0.1)
        pa = st.number_input("Prealbumin (200–400 mg/L)", min_value=0.0, max_value=1000.0, step=1.0)
        pct = st.number_input("Plateletcrit (0.108–0.282%)", min_value=0.0, max_value=1.0, step=0.001)
        tp = st.number_input("Total Protein (60–83 g/L)", min_value=0.0, max_value=150.0, step=0.1)

    # Prepare input data
    input_data = [gender, alb, cl, cr, glb, hdl, na, pa, pct, tp]

    # Prediction button and results
    if st.button("Predict Diagnosis", key="predict_button"):
        if model is None:
            st.error("Model not loaded. Please check the model file.")
        elif any(v == 0 or v is None for v in input_data[1:]):  # Allow gender to be 0
            st.warning("Please fill all fields with valid values before predicting.")
        else:
            diagnosis, confidence = predict_diagnosis(input_data)
            if diagnosis:
                color = "#C00000" if diagnosis == "MTB" else "#C00000"
                st.markdown(
                    f"""
                    <div style='background-color: #f0f0f0; padding: 20px; border-radius: 20px; text-align: center;'>
                        <h2 style='color: {color};'>Predicted Diagnosis: {diagnosis}</h2>
                        <p style='font-size: 20px; color: #C00000;'>Accuracy: {confidence:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Add sidebar with additional information
    with st.sidebar:
        st.header("Reference Ranges")
        st.markdown("""
        - Gender: Male (0), Female (1)
        - Albumin: 35–50 g/L
        - Chloride: 98–106 mmol/L
        - Creatinine: 
          - Male: 59–104 μmol/L
          - Female: 45–84 μmol/L
        - Globulin: 20-30 g/L
        - HDL: 1.04–1.55 mmol/L
        - Sodium: 135–145 mmol/L
        - Prealbumin: 200–400 mg/L
        - Plateletcrit: 0.108–0.282%
        - Total Protein: 60–83 g/L
        """)

if __name__ == '__main__':
    if model:
        main()
    else:
        st.error("Application cannot start due to missing model.")
