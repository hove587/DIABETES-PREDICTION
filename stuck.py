import streamlit as st
import pandas as pd
import numpy as np

# Check for required packages and install if missing
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.info("Please run: pip install scikit-learn")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# App title
st.title("🩺 Diabetes Risk Prediction App")
st.write("Check your diabetes risk using easily measurable health factors")

# Sidebar
with st.sidebar:
    st.header("About This App")
    st.write("""
    This app predicts diabetes risk based on health factors.
    **Note:** For educational purposes only.
    Consult healthcare professionals for medical advice.
    """)

# Load data function
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(url, names=column_names)
        
        # Clean data
        df_clean = df.drop('SkinThickness', axis=1)
        medical_columns = ['Glucose', 'BloodPressure', 'Insulin', 'BMI']
        for col in medical_columns:
            df_clean[col] = df_clean[col].replace(0, np.nan)
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        return df_clean
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Train model function
@st.cache_resource
def train_model():
    df_clean = load_data()
    if df_clean is None:
        return None
        
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    return model

# Main app
def main():
    # Load model
    with st.spinner('Loading prediction model...'):
        model = train_model()
    
    if model is None:
        st.error("Failed to load the prediction model. Please check your internet connection and try again.")
        return

    # Input form
    st.header("Enter Your Health Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age (years)", 20, 80, 40)
        pregnancies = st.slider("Number of Pregnancies", 0, 10, 0)
        glucose = st.slider("Glucose Level (mg/dL)", 70, 200, 100)
        
    with col2:
        bmi = st.slider("BMI", 15.0, 40.0, 25.0, 0.1)
        bp_diastolic = st.slider("Diastolic Blood Pressure (mmHg)", 60, 120, 80)
        insulin = st.slider("Insulin Level", 20, 300, 100)
        family_history = st.slider("Family History Strength", 0.0, 2.5, 0.5, 0.1)

    # Prediction
    if st.button("Calculate Diabetes Risk", type="primary"):
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [bp_diastolic],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [family_history],
            'Age': [age]
        })
        
        try:
            probability = model.predict_proba(input_data)[0][1]
            
            # Display results
            st.success("Prediction Complete!")
            
            # Risk level
            if probability < 0.25:
                risk_level = "LOW"
                color = "green"
                advice = "Maintain your healthy lifestyle!"
            elif probability < 0.5:
                risk_level = "MODERATE" 
                color = "orange"
                advice = "Consider lifestyle improvements."
            elif probability < 0.75:
                risk_level = "HIGH"
                color = "red"
                advice = "Consult a healthcare provider."
            else:
                risk_level = "VERY HIGH"
                color = "darkred"
                advice = "See a doctor soon."
            
            # Results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Level", risk_level)
            with col2:
                st.metric("Probability", f"{probability*100:.1f}%")
            with col3:
                st.metric("Recommendation", advice)
            
            # Risk factors
            st.subheader("Identified Risk Factors")
            factors = []
            if glucose >= 126: factors.append(f"High glucose ({glucose} mg/dL)")
            elif glucose >= 100: factors.append(f"Elevated glucose ({glucose} mg/dL)")
            if bmi >= 30: factors.append(f"High BMI ({bmi})")
            elif bmi >= 25: factors.append(f"Elevated BMI ({bmi})")
            if age > 45: factors.append("Age above 45")
            if bp_diastolic >= 90: factors.append(f"High blood pressure ({bp_diastolic} mmHg)")
            if family_history > 1.0: factors.append("Strong family history")
            
            if factors:
                for factor in factors:
                    st.write(f"• {factor}")
            else:
                st.write("• No major risk factors identified")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")

# BMI Calculator
st.header("BMI Calculator")
weight_col, height_col, result_col = st.columns(3)

with weight_col:
    weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
with height_col:
    height = st.number_input("Height (m)", 1.4, 2.2, 1.7, 0.01)
with result_col:
    if weight > 0 and height > 0:
        bmi_calc = weight / (height ** 2)
        st.metric("Your BMI", f"{bmi_calc:.1f}")

if __name__ == "__main__":
    main()
