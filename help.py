import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low { color: green; font-weight: bold; }
    .risk-moderate { color: orange; font-weight: bold; }
    .risk-high { color: red; font-weight: bold; }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🩺 Diabetes Risk Prediction App</h1>', unsafe_allow_html=True)

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About This App")
    st.write("""
    This app predicts your risk of developing diabetes based on 
    easily measurable health factors.
    
    **How it works:**
    - Enter your health information
    - Get instant risk assessment
    - Receive personalized health tips
    
    **Note:** This is for educational purposes only. 
    Always consult healthcare professionals for medical advice.
    """)
    
    st.header("📊 Dataset Info")
    st.write("""
    - **Data Source:** Pima Indians Diabetes Dataset
    - **Samples:** 768 people
    - **Accuracy:** 75-80%
    - **Factors:** Glucose, BMI, Age, Blood Pressure, etc.
    """)

# Load and prepare data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=column_names)
    
    # Remove SkinThickness and clean data
    df_clean = df.drop('SkinThickness', axis=1)
    medical_columns = ['Glucose', 'BloodPressure', 'Insulin', 'BMI']
    for col in medical_columns:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    return df_clean

@st.cache_resource
def train_model():
    df_clean = load_data()
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    return model

# Load data and model
df_clean = load_data()
model = train_model()

# Main app content
tab1, tab2, tab3 = st.tabs(["🎯 Risk Assessment", "📊 Health Calculator", "📚 Learn More"])

with tab1:
    st.header("Personal Diabetes Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        age = st.slider("Age (years)", 20, 80, 40)
        pregnancies = st.slider("Number of Pregnancies", 0, 10, 0)
        
        st.subheader("Health Measurements")
        glucose = st.slider("Glucose Level (mg/dL)", 70, 200, 100)
        bmi = st.slider("BMI (Body Mass Index)", 15.0, 40.0, 25.0, 0.1)
    
    with col2:
        st.subheader("Additional Factors")
        bp_diastolic = st.slider("Diastolic Blood Pressure (mmHg)", 60, 120, 80)
        insulin = st.slider("Insulin Level (mu U/ml)", 20, 300, 100)
        diabetes_pedigree = st.slider("Family History Strength (0-2.5)", 0.0, 2.5, 0.5, 0.1)
    
    # Prediction button
    if st.button("🔍 Calculate My Diabetes Risk", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [bp_diastolic],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age]
        })
        
        # Make prediction
        probability = model.predict_proba(input_data)[0][1]
        
        # Display results
        st.markdown("---")
        st.header("📊 Your Diabetes Risk Report")
        
        # Risk level determination
        if probability < 0.25:
            risk_level = "LOW"
            risk_class = "risk-low"
            emoji = "🟢"
            advice = "Maintain your healthy lifestyle!"
        elif probability < 0.5:
            risk_level = "MODERATE"
            risk_class = "risk-moderate"
            emoji = "🟡"
            advice = "Consider lifestyle improvements and monitor your health."
        elif probability < 0.75:
            risk_level = "HIGH"
            risk_class = "risk-high"
            emoji = "🟠"
            advice = "Recommended to consult with a healthcare provider."
        else:
            risk_level = "VERY HIGH"
            risk_class = "risk-high"
            emoji = "🔴"
            advice = "Strongly recommended to see a doctor soon."
        
        # Results columns
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric("Risk Level", f"{emoji} {risk_level}")
        with res_col2:
            st.metric("Probability", f"{probability*100:.1f}%")
        with res_col3:
            st.metric("Recommendation", advice)
        
        # Risk factors analysis
        st.subheader("⚠️ Identified Risk Factors")
        risk_factors = []
        
        if glucose >= 126:
            risk_factors.append(f"High glucose level ({glucose} mg/dL - diabetic range)")
        elif glucose >= 100:
            risk_factors.append(f"Elevated glucose level ({glucose} mg/dL - pre-diabetic)")
            
        if bmi >= 30:
            risk_factors.append(f"High BMI ({bmi} - obese range)")
        elif bmi >= 25:
            risk_factors.append(f"Elevated BMI ({bmi} - overweight range)")
            
        if age > 45:
            risk_factors.append(f"Age above 45")
            
        if bp_diastolic >= 90:
            risk_factors.append(f"High diastolic blood pressure ({bp_diastolic} mmHg)")
        elif bp_diastolic >= 80:
            risk_factors.append(f"Elevated diastolic blood pressure ({bp_diastolic} mmHg)")
            
        if diabetes_pedigree > 1.0:
            risk_factors.append("Strong family history of diabetes")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.write("• No major risk factors identified")
        
        # Health tips
        st.subheader("💪 Personalized Health Tips")
        if glucose > 100:
            st.write("• Reduce sugar and refined carbohydrate intake")
            st.write("• Increase physical activity")
        if bmi > 25:
            st.write("• Aim for gradual weight loss through diet and exercise")
            st.write("• Include more fruits and vegetables in your diet")
        if bp_diastolic > 80:
            st.write("• Reduce salt intake")
            st.write("• Practice stress management techniques")
        if age > 40:
            st.write("• Schedule regular health checkups")

with tab2:
    st.header("Health Calculators")
    
    calc_option = st.radio("Choose Calculator:", ["BMI Calculator", "Blood Pressure Guide"])
    
    if calc_option == "BMI Calculator":
        st.subheader("BMI Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0, 0.1)
        with col2:
            height = st.number_input("Height (meters)", 1.4, 2.2, 1.7, 0.01)
        
        if weight > 0 and height > 0:
            bmi_calculated = weight / (height ** 2)
            
            if bmi_calculated < 18.5:
                category = "Underweight"
                color = "blue"
            elif bmi_calculated < 25:
                category = "Normal weight"
                color = "green"
            elif bmi_calculated < 30:
                category = "Overweight"
                color = "orange"
            else:
                category = "Obese"
                color = "red"
            
            st.metric("Your BMI", f"{bmi_calculated:.1f} ({category})")
            
            st.progress(min(bmi_calculated / 40, 1.0))
            st.write("**BMI Categories:**")
            st.write("• Underweight: < 18.5")
            st.write("• Normal weight: 18.5 - 24.9")
            st.write("• Overweight: 25 - 29.9")
            st.write("• Obese: 30 or higher")
    
    else:
        st.subheader("Blood Pressure Guide")
        
        st.write("""
        **Understanding Blood Pressure Readings:**
        
        Blood pressure has two numbers:
        - **Systolic (Top):** Pressure when heart beats
        - **Diastolic (Bottom):** Pressure between beats
        
        **Categories:**
        - **Normal:** Below 120/80 mmHg
        - **Elevated:** 120-129/Below 80 mmHg
        - **High Stage 1:** 130-139/80-89 mmHg
        - **High Stage 2:** 140+/90+ mmHg
        
        **Note:** This app uses **diastolic pressure** (bottom number) for predictions.
        """)

with tab3:
    st.header("Learn About Diabetes Risk Factors")
    
    st.write("""
    ## 🔍 Understanding Diabetes Risk Factors
    
    **Glucose (Blood Sugar):**
    - Normal: Below 100 mg/dL
    - Pre-diabetes: 100-125 mg/dL
    - Diabetes: 126 mg/dL or higher
    
    **BMI (Body Mass Index):**
    - Underweight: Below 18.5
    - Healthy: 18.5 - 24.9
    - Overweight: 25 - 29.9
    - Obese: 30 or higher
    
    **Age:**
    - Risk increases after age 45
    - Regular screening recommended after 40
    
    **Blood Pressure:**
    - Normal: Below 120/80 mmHg
    - High: 130/80 mmHg or higher
    
    **Family History:**
    - Higher risk if parents or siblings have diabetes
    """)
    
    st.markdown("---")
    st.info("""
    **Important Disclaimer:** 
    This application is for educational and awareness purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of your physician or other qualified health provider with 
    any questions you may have regarding a medical condition.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Diabetes Prediction App | Educational Purpose Only | "
    "Data Source: Pima Indians Diabetes Dataset"
    "</div>", 
    unsafe_allow_html=True
)
