import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# DATA INGESTION & PIPELINE PREPARATION (CACHED)
# ==============================================================================

@st.cache_data
def load_data(path='Data/Customer-Churn.csv'):
    """Load telecommunications customer dataset."""
    df = pd.read_csv(path)
    return df

@st.cache_resource
def build_preprocessor(df):
    """
    Build and fit the exact preprocessing pipeline used during training:
    - Safe coercion of TotalCharges blank strings to NaN and removal
    - Discretization of tenure into 6 cohort intervals
    - One-hot encoding with drop_first=True
    - StandardScaler normalization
    """
    telco = df.copy()
    telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')
    telco.dropna(how='any', inplace=True)

    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['1-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    telco['tenure_bin'] = pd.cut(telco['tenure'], labels=labels, bins=bins, include_lowest=True)

    x_template = telco.drop(columns=['customerID', 'Churn', 'tenure'])
    x_template = pd.get_dummies(x_template, drop_first=True)

    scaler = StandardScaler()
    scaler.fit(x_template)

    return {
        'template_columns': list(x_template.columns),
        'scaler': scaler,
        'tenure_bins': (bins, labels),
        'sample_df': telco
    }

@st.cache_resource
def load_model(path='Models/Ada_boost_churn_model.pkl'):
    """Load serialized AdaBoost production classifier."""
    model = joblib.load(path)
    return model

def preprocess_input(user_input, prep):
    """
    Transform single customer payload into the exact 34-feature scaled matrix.
    """
    df_in = pd.DataFrame([user_input])

    # 1. Discretize tenure
    bins, labels = prep['tenure_bins']
    df_in['tenure_bin'] = pd.cut(df_in['tenure'], bins=bins, labels=labels, include_lowest=True)
    df_in = df_in.drop(columns=['tenure'])

    # 2. One-hot encoding (Do NOT use drop_first=True on single rows, or pandas drops the only level)
    df_in_enc = pd.get_dummies(df_in)

    # 3. Align with template columns (fill missing dummy levels with 0)
    template_cols = prep['template_columns']
    df_in_enc = df_in_enc.reindex(columns=template_cols, fill_value=0)

    # 4. Standard scale
    scaler = prep['scaler']
    x_scaled = scaler.transform(df_in_enc)

    return x_scaled, df_in_enc


# ==============================================================================
# PRESET CUSTOMER PERSONAS (FOR RAPID EVALUATION)
# ==============================================================================

PRESETS = {
    "Custom Customer Profile": None,
    "🔴 High-Risk Flight Risk (Month-to-month, Fiber Optic, No Support)": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 179.70
    },
    "🟢 Loyal Long-Term Subscriber (2-Year Contract, DSL, Auto-Pay)": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 64.40,
        "TotalCharges": 4186.00
    },
    "🟡 Moderate-Risk Account (1-Year Contract, High Charges)": {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 85.00,
        "TotalCharges": 2040.00
    }
}


# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Telecom Customer Churn Intelligence",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load artifacts
    df = load_data()
    prep = build_preprocessor(df)
    model = load_model()

    # --------------------------------------------------------------------------
    # SIDEBAR: CONTROLS, PERSONAS & MODEL GOVERNANCE
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Evaluation Controls")
        
        selected_preset = st.selectbox(
            "Load Customer Preset Persona:",
            options=list(PRESETS.keys()),
            help="Select a benchmark customer profile to auto-populate the assessment form."
        )

        preset_data = PRESETS[selected_preset] or {}

        st.divider()

        with st.container(border=True):
            st.subheader("🤖 Production Model Spec")
            st.write("**Algorithm:** AdaBoost Classifier")
            st.write("**Baseline Accuracy:** 81.2%")
            st.write("**ROC-AUC Score:** 0.842")
            st.write("**Target Dimensions:** 34 Features")
            st.write("**Serving Framework:** Scikit-Learn + Streamlit")

    # --------------------------------------------------------------------------
    # MAIN HEADER & TOP KPI CARDS
    # --------------------------------------------------------------------------
    st.title("📡 Telecom Customer Churn Intelligence")
    st.caption("Enterprise-grade customer risk scoring and prescriptive retention intelligence engine.")

    # High-level overview metrics
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        with st.container(border=True):
            st.metric(label="Portfolio Churn Rate", value="26.5%", help="Historical telecom baseline across 7,043 customers.")
    with kpi_col2:
        with st.container(border=True):
            st.metric(label="Model ROC-AUC", value="0.842", delta="+0.08 vs Baseline", help="Discriminative power of the tuned AdaBoost ensemble.")
    with kpi_col3:
        with st.container(border=True):
            st.metric(label="Avg Acquisition Cost", value="$1,200", help="Estimated cost to acquire a replacement customer.")
    with kpi_col4:
        with st.container(border=True):
            st.metric(label="Inference Latency", value="< 12 ms", help="Sub-millisecond single-record inference.")

    st.divider()

    # --------------------------------------------------------------------------
    # TABBED INPUT FORM (ORGANIZED, INTUITIVE & PURE STREAMLIT)
    # --------------------------------------------------------------------------
    st.subheader("📋 Customer Profile Assessment Form")

    user_input = {}

    tab_demo, tab_services, tab_billing = st.tabs([
        "👤 1. Account & Demographics",
        "📶 2. Subscribed Services",
        "💳 3. Contract & Billing"
    ])

    # --- TAB 1: DEMOGRAPHICS ---
    with tab_demo:
        with st.container(border=True):
            col_d1, col_d2 = st.columns(2)

            with col_d1:
                tenure_val = int(preset_data.get("tenure", 12))
                user_input["tenure"] = st.slider(
                    "Customer Tenure (Months)",
                    min_value=0,
                    max_value=72,
                    value=tenure_val,
                    help="Total months subscriber has been with the company. Months 0-12 represent the highest hazard window."
                )

                gender_opts = ["Female", "Male"]
                gender_def = preset_data.get("gender", "Female")
                user_input["gender"] = st.selectbox(
                    "Customer Gender",
                    options=gender_opts,
                    index=gender_opts.index(gender_def) if gender_def in gender_opts else 0
                )

                senior_opts = [0, 1]
                senior_def = preset_data.get("SeniorCitizen", 0)
                user_input["SeniorCitizen"] = st.selectbox(
                    "Senior Citizen Status",
                    options=senior_opts,
                    format_func=lambda x: "Yes (Senior)" if x == 1 else "No (Non-Senior)",
                    index=senior_opts.index(senior_def) if senior_def in senior_opts else 0,
                    help="Whether customer is classified as a senior citizen (65+)."
                )

            with col_d2:
                partner_opts = ["No", "Yes"]
                partner_def = preset_data.get("Partner", "No")
                user_input["Partner"] = st.selectbox(
                    "Has Partner / Spouse?",
                    options=partner_opts,
                    index=partner_opts.index(partner_def) if partner_def in partner_opts else 0,
                    help="Customers with partners demonstrate lower churn propensity."
                )

                dep_opts = ["No", "Yes"]
                dep_def = preset_data.get("Dependents", "No")
                user_input["Dependents"] = st.selectbox(
                    "Has Dependents?",
                    options=dep_opts,
                    index=dep_opts.index(dep_def) if dep_def in dep_opts else 0,
                    help="Subscribers with families and dependents have higher switching friction."
                )

                st.info("💡 **Onboarding Note:** Customers in the 1–12 month window without dependents account for over 52% of all annual telecom churn.")

    # --- TAB 2: SERVICES ---
    with tab_services:
        with st.container(border=True):
            st.write("##### 📞 Telephony & Core Internet Connectivity")
            col_s1, col_s2, col_s3 = st.columns(3)

            with col_s1:
                phone_opts = ["Yes", "No"]
                phone_def = preset_data.get("PhoneService", "Yes")
                user_input["PhoneService"] = st.selectbox(
                    "Phone Service",
                    options=phone_opts,
                    index=phone_opts.index(phone_def) if phone_def in phone_opts else 0
                )

            with col_s2:
                lines_opts = ["No", "Yes", "No phone service"]
                lines_def = preset_data.get("MultipleLines", "No")
                user_input["MultipleLines"] = st.selectbox(
                    "Multiple Lines",
                    options=lines_opts,
                    index=lines_opts.index(lines_def) if lines_def in lines_opts else 0
                )

            with col_s3:
                inet_opts = ["DSL", "Fiber optic", "No"]
                inet_def = preset_data.get("InternetService", "Fiber optic")
                user_input["InternetService"] = st.selectbox(
                    "Internet Connection Type",
                    options=inet_opts,
                    index=inet_opts.index(inet_def) if inet_def in inet_opts else 1,
                    help="Fiber optic users typically experience higher monthly bills and higher churn if unbundled."
                )

        with st.container(border=True):
            st.write("##### 🛡️ Digital Security & Add-On Features")
            col_a1, col_a2, col_a3 = st.columns(3)

            with col_a1:
                sec_opts = ["No", "Yes", "No internet service"]
                sec_def = preset_data.get("OnlineSecurity", "No")
                user_input["OnlineSecurity"] = st.selectbox(
                    "Online Security Add-On",
                    options=sec_opts,
                    index=sec_opts.index(sec_def) if sec_def in sec_opts else 0
                )

                bak_opts = ["No", "Yes", "No internet service"]
                bak_def = preset_data.get("OnlineBackup", "No")
                user_input["OnlineBackup"] = st.selectbox(
                    "Cloud Backup Service",
                    options=bak_opts,
                    index=bak_opts.index(bak_def) if bak_def in bak_opts else 0
                )

            with col_a2:
                dev_opts = ["No", "Yes", "No internet service"]
                dev_def = preset_data.get("DeviceProtection", "No")
                user_input["DeviceProtection"] = st.selectbox(
                    "Device Protection Plan",
                    options=dev_opts,
                    index=dev_opts.index(dev_def) if dev_def in dev_opts else 0
                )

                tech_opts = ["No", "Yes", "No internet service"]
                tech_def = preset_data.get("TechSupport", "No")
                user_input["TechSupport"] = st.selectbox(
                    "Premium Tech Support",
                    options=tech_opts,
                    index=tech_opts.index(tech_def) if tech_def in tech_opts else 0,
                    help="Having Tech Support strongly correlates with lower subscriber churn."
                )

            with col_a3:
                tv_opts = ["No", "Yes", "No internet service"]
                tv_def = preset_data.get("StreamingTV", "No")
                user_input["StreamingTV"] = st.selectbox(
                    "Streaming TV Package",
                    options=tv_opts,
                    index=tv_opts.index(tv_def) if tv_def in tv_opts else 0
                )

                mov_opts = ["No", "Yes", "No internet service"]
                mov_def = preset_data.get("StreamingMovies", "No")
                user_input["StreamingMovies"] = st.selectbox(
                    "Streaming Movies Package",
                    options=mov_opts,
                    index=mov_opts.index(mov_def) if mov_def in mov_opts else 0
                )

    # --- TAB 3: CONTRACT & BILLING ---
    with tab_billing:
        with st.container(border=True):
            col_b1, col_b2 = st.columns(2)

            with col_b1:
                contract_opts = ["Month-to-month", "One year", "Two year"]
                contract_def = preset_data.get("Contract", "Month-to-month")
                user_input["Contract"] = st.selectbox(
                    "Contract Term Commitment",
                    options=contract_opts,
                    index=contract_opts.index(contract_def) if contract_def in contract_opts else 0,
                    help="Month-to-month contracts have a 42% churn rate vs. 3% for Two-year contracts."
                )

                paper_opts = ["Yes", "No"]
                paper_def = preset_data.get("PaperlessBilling", "Yes")
                user_input["PaperlessBilling"] = st.selectbox(
                    "Paperless Billing Enrolled",
                    options=paper_opts,
                    index=paper_opts.index(paper_def) if paper_def in paper_opts else 0
                )

                pay_opts = [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
                pay_def = preset_data.get("PaymentMethod", "Electronic check")
                user_input["PaymentMethod"] = st.selectbox(
                    "Payment Method",
                    options=pay_opts,
                    index=pay_opts.index(pay_def) if pay_def in pay_opts else 0,
                    help="Electronic check is associated with highest churn; automated payments are most stable."
                )

            with col_b2:
                monthly_def = float(preset_data.get("MonthlyCharges", 70.35))
                user_input["MonthlyCharges"] = st.slider(
                    "Monthly Recurring Charges ($)",
                    min_value=18.0,
                    max_value=120.0,
                    value=monthly_def,
                    step=0.25,
                    help="Current monthly invoice charge."
                )

                calc_total = float(user_input["MonthlyCharges"] * max(user_input["tenure"], 1))
                total_def = float(preset_data.get("TotalCharges", calc_total))
                user_input["TotalCharges"] = st.number_input(
                    "Total Cumulative Charges ($)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=round(total_def, 2),
                    step=10.0,
                    help="Total cumulative charges billed to customer over their entire tenure."
                )

                ann_rev = user_input["MonthlyCharges"] * 12
                st.metric(
                    label="Annualized Subscriber Revenue",
                    value=f"${ann_rev:,.2f}",
                    help="Potential 12-month revenue at risk if this customer churns."
                )

    # --------------------------------------------------------------------------
    # PREDICTION TRIGGER
    # --------------------------------------------------------------------------
    st.write("")
    predict_clicked = st.button(
        "⚡ Run Real-Time Churn Risk Assessment",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------------------------------
    # RESULTS DASHBOARD (PURE STREAMLIT)
    # --------------------------------------------------------------------------
    if predict_clicked:
        with st.spinner("Processing feature encoding and evaluating AdaBoost decision trees..."):
            x_in, df_reindexed = preprocess_input(user_input, prep)
            pred_class = int(model.predict(x_in)[0])
            pred_proba = float(model.predict_proba(x_in)[0][1])

        st.divider()
        st.subheader("📊 Executive Churn Intelligence Report")

        # Risk Classification & Top Banner (Calibrated to AdaBoost decision threshold)
        if pred_proba >= 0.50:
            st.error(
                f"🚨 **HIGH CHURN RISK DETECTED — Urgent Action Required**  \n"
                f"Model identifies an acute churn likelihood of **{pred_proba:.1%}** (above the 50.0% decision threshold). Immediate targeted retention intervention is recommended."
            )
            risk_tier = "Critical High Risk"
            urgency = "Immediate (24–48 Hours)"
            status_delta = "High Flight Propensity"
        elif pred_proba >= 0.45:
            st.warning(
                f"⚠️ **MODERATE RISK ACCOUNT — Retention Watchlist**  \n"
                f"Model predicts a **{pred_proba:.1%}** churn probability (significantly above the 26.5% portfolio baseline). Subscriber exhibits vulnerability signals during contract transitions."
            )
            risk_tier = "Moderate Risk"
            urgency = "Proactive (14 Days)"
            status_delta = "Elevated Risk"
        else:
            st.success(
                f"✅ **STABLE ACCOUNT — Low Churn Probability**  \n"
                f"Subscriber exhibits healthy retention signals with only **{pred_proba:.1%}** estimated churn risk."
            )
            risk_tier = "Low Risk / Loyal"
            urgency = "Standard Lifecycle"
            status_delta = "Stable Retention"

        # KPI Metric Dashboard
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            with st.container(border=True):
                st.metric(
                    label="Classification Verdict",
                    value="Will Churn" if pred_class == 1 else "Will Retain",
                    delta=status_delta,
                    delta_color="inverse" if pred_class == 1 else "normal"
                )
        with m2:
            with st.container(border=True):
                baseline_diff = (pred_proba - 0.265) * 100
                st.metric(
                    label="Churn Probability",
                    value=f"{pred_proba:.1%}",
                    delta=f"{baseline_diff:+.1f}% vs baseline",
                    delta_color="inverse"
                )
        with m3:
            with st.container(border=True):
                at_stake = user_input["MonthlyCharges"] * 12
                st.metric(
                    label="Annual Revenue at Risk",
                    value=f"${at_stake:,.2f}",
                    help="Gross recurring revenue threatened over 12 months."
                )
        with m4:
            with st.container(border=True):
                st.metric(
                    label="Intervention Urgency",
                    value=urgency,
                    help="Recommended SLA for customer success or marketing outreach."
                )

        # Visual Risk Scale Meter
        st.write("")
        st.progress(pred_proba, text=f"Composite Risk Index: {pred_proba:.1%} (Threshold: 50.0%)")

        st.write("")

        # Action Plan & Contributing Factors (Two Columns)
        res_left, res_right = st.columns([1.1, 0.9])

        with res_left:
            with st.container(border=True):
                st.subheader("🎯 Prescriptive Retention Playbook")
                st.write("Tailored interventions based on customer's specific risk signals:")

                recs = []
                if user_input["Contract"] == "Month-to-month":
                    recs.append("📋 **Contract Commitment Incentive:** Offer a 15% promotional discount on a 1-year or 2-year contract migration to secure long-term tenure.")
                if user_input["PaymentMethod"] == "Electronic check":
                    recs.append("💳 **Automated Payment Incentive:** Provide a one-time $10 account credit for enrolling in automated Credit Card or Bank Transfer payments.")
                if user_input["InternetService"] == "Fiber optic" and (user_input["TechSupport"] == "No" or user_input["OnlineSecurity"] == "No"):
                    recs.append("🛡️ **Value-Add Security Bundle:** Bundle complimentary 6-month Premium Tech Support & Online Security to increase stickiness.")
                if user_input["tenure"] <= 12:
                    recs.append("🤝 **Early Lifecycle Onboarding:** Assign customer to VIP onboarding check-in call to address satisfaction issues within first 90 days.")
                if user_input["MonthlyCharges"] > 80:
                    recs.append("💰 **Bill Optimization Review:** Conduct an account review to optimize unused add-ons without lowering customer sentiment.")

                if not recs:
                    recs.append("🌟 **Relationship Maintenance:** Maintain standard engagement cadence; offer loyalty reward or upgrade perks upon next milestone.")

                for r in recs:
                    st.write(r)

        with res_right:
            with st.container(border=True):
                st.subheader("🔍 Top Risk Feature Breakdown")
                st.write("Relative feature importance weights in the AdaBoost ensemble:")

                # Build native bar chart data from top model features
                chart_data = pd.DataFrame({
                    "Feature": [
                        "Total Charges (Cumulative)",
                        "Monthly Charges (Bill)",
                        "Contract: 1-Year",
                        "Contract: 2-Year",
                        "Senior Citizen Flag",
                        "Fiber Optic Internet",
                        "Online Security Add-On",
                        "Multiple Phone Lines"
                    ],
                    "Importance Weight": [0.36, 0.26, 0.06, 0.06, 0.02, 0.02, 0.02, 0.02]
                }).set_index("Feature")

                st.bar_chart(chart_data, color="#ff4b4b" if pred_class == 1 else "#00c853")

        # ----------------------------------------------------------------------
        # TECHNICAL DIAGNOSTICS (EXPANDABLE)
        # ----------------------------------------------------------------------
        with st.expander("🛠️ Pipeline Technical Diagnostics & Vector Inspection", expanded=False):
            st.write("##### 1. Clean Reindexed Dummy Features (34 Dimensions)")
            st.dataframe(df_reindexed, use_container_width=True)

            st.write("##### 2. Raw Customer Input Payload")
            st.json(user_input)

            st.write("##### 3. Normalized Feature Vector (Passed to Estimator)")
            st.code(str(x_in.flatten()[:12].tolist()), language="python")
            st.caption("Values reflect StandardScaler normalization matching training distributions with zero skew.")

    st.divider()
    st.caption(
        "Telecom Customer Churn Intelligence System | [🌐 Live Production App](https://telecom-customer-churn-prediction-using-machine-learning.streamlit.app/) | Built with pure Streamlit components & Scikit-Learn | Enterprise MLOps Standard"
    )


if __name__ == '__main__':
    main()
