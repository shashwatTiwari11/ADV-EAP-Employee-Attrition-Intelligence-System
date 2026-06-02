import streamlit as st
import requests
import pandas as pd
import time

API_URL = "http://127.0.0.1:8000"



st.set_page_config(
page_title="EAP Attrition Intelligence",
page_icon="🏢",
layout="wide"
)



with st.sidebar:
    st.title("🏢 HR Intelligence")
    st.markdown("### Navigation")


    page = st.radio(
        "Go to",
        ["🧍 Individual Prediction", "📂 Batch Prediction"]
    )

    st.markdown("---")
    st.success("Model: Production")
    st.caption("AI Decision System")




st.markdown("""

<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
    url("https://images.unsplash.com/photo-1497366754035-f200968a6e72");
    background-size: cover;
}
h1, h2, h3, h4, h5, h6, p, label {
    color: white;
}
</style>

""", unsafe_allow_html=True)



if page == "🧍 Individual Prediction":


 st.title("Employee Attrition Risk Monitor")
st.caption("Real-time predictive intelligence for HR teams")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Employee Profile")
    Age = st.number_input("Age", 18, 60, 30)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )
    JobRole = st.selectbox(
        "Job Role",
        [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative",
            "Manager", "Sales Representative", "Research Director", "Human Resources"
        ]
    )
    MonthlyIncome = st.number_input("Monthly Income", 1000, 500000, 5000)
    OverTime = st.selectbox("Overtime", ["Yes", "No"])

with col2:
    st.subheader("Engagement & Satisfaction")
    JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    WorkLifeBalance = st.slider("Work-Life Balance", 1, 4, 3)
    TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 5)
    YearsAtCompany = st.number_input("Years At Company", 0, 40, 3)

st.markdown("---")


payload = {
    "Age": Age,
    "BusinessTravel": "Travel_Rarely",
    "Department": Department,
    "DistanceFromHome": 5,
    "Education": 3,
    "EnvironmentSatisfaction": EnvironmentSatisfaction,
    "Gender": Gender,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": JobRole,
    "JobSatisfaction": JobSatisfaction,
    "MonthlyIncome": MonthlyIncome,
    "OverTime": OverTime,
    "TotalWorkingYears": TotalWorkingYears,
    "YearsAtCompany": YearsAtCompany,
    "WorkLifeBalance": WorkLifeBalance
}

response = requests.post(f"{API_URL}/predict", json=payload)

if response.status_code == 200:
    result = response.json()
    probability = float(result["probability_of_attrition"])

    st.markdown("## 📊 Risk Analysis")
    st.progress(int(probability * 100))

    if probability < 0.35:
        st.success(f"LOW RISK 🟢 ({probability:.1%})")
    elif probability < 0.65:
        st.warning(f"MODERATE RISK 🟠 ({probability:.1%})")
    else:
        st.error(f"HIGH RISK 🔴 ({probability:.1%})")

    st.markdown("### 💡 Recommendation")

    if probability >= 0.65:
        st.error("Immediate intervention required")
    elif probability >= 0.35:
        st.warning("Monitor engagement closely")
    else:
        st.success("Employee is stable")




if page == "📂 Batch Prediction":


 st.title("Batch Attrition Analysis")
st.caption("Upload dataset for bulk prediction")
st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df_preview = pd.read_csv(uploaded_file)
    st.markdown("### 🔍 Data Preview")
    st.dataframe(df_preview.head(), use_container_width=True)

    uploaded_file.seek(0)

    with st.spinner("Processing dataset..."):

        response = requests.post(
            f"{API_URL}/batch_predict",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv"
                )
            }
        )

    if response.status_code == 200:
        result_df = pd.DataFrame(response.json())

        st.markdown("## 📊 Results Overview")

        total = len(result_df)
        attrition = int(result_df["prediction"].sum())

        col1, col2 = st.columns(2)
        col1.metric("Total Employees", total)
        col2.metric("Attrition Predictions", attrition)

        st.bar_chart(result_df["prediction"].value_counts())

        st.success("✅ Batch prediction completed")

 

        # Top 10
        top_risk = result_df.sort_values(
            by="probability_of_attrition", ascending=False
        ).head(10)

        st.markdown("### 🔝 Top High-Risk Employees")
        st.dataframe(top_risk, use_container_width=True)

        # Risk segmentation
        def risk_category(p):
            if p < 0.35:
                return "Low"
            elif p < 0.65:
                return "Medium"
            else:
                return "High"

        result_df["risk_category"] = result_df["probability_of_attrition"].apply(risk_category)

        st.markdown("### 📊 Risk Distribution")
        st.bar_chart(result_df["risk_category"].value_counts())

        # Simple insights
        st.markdown("### 💡 Key Insights")

        if "OverTime" in result_df.columns:
            high_ot = result_df[result_df["OverTime"] == "Yes"]["probability_of_attrition"].mean()
            low_ot = result_df[result_df["OverTime"] == "No"]["probability_of_attrition"].mean()

            if high_ot > low_ot:
                st.warning("Employees working overtime show higher attrition risk")

        # Actions
        st.markdown("### 🎯 HR Action")

        high_count = len(result_df[result_df["risk_category"] == "High"])

        if high_count > total * 0.3:
            st.error("High attrition risk — Immediate action required")
        elif high_count > total * 0.15:
            st.warning("Moderate risk — Monitor employees")
        else:
            st.success("Attrition under control")

        # Download
        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Results",
            data=csv,
            file_name="attrition_predictions.csv",
            mime="text/csv"
        )

    else:
        st.error("Batch API failed")
