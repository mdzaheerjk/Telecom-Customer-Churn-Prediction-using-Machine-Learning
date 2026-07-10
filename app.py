import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data(path='Data/Customer-Churn.csv'):
    df=pd.read_csv(path)
    return df

@st.cache_resource
def build_preprocessor(df):
    telco=df.copy()
    telco['TotalCharges']=pd.to_numeric(telco['TotalCharges'],errors='coerce')
    telco.dropna(how='any',inplace=True)

    bins=[0,12,24,36,48,60,72]
    labels=['1-12','13-24','25-36','37-48','49-60','61-72']
    telco['tenure_bin']=pd.cut(telco['tenure'],labels=labels,bins=bins,include_lowest=True)

    x_template=telco.drop(columns=['customerID','Churn','tenure'])
    x_template=pd.get_dummies(x_template,drop_first=True)

    scaler=StandardScaler()
    scaler.fit(x_template)

    print(f"[DEBUG] build_preprocessor : template_cols={x_template.shape[1]},template_rows={x_template.shape[0]}")

    return {
        'template_columns':list(x_template.columns),
        'scaler':scaler,
        'tenure_bins':(bins,labels),
        'sample_df':telco
    }

def preprocessor_input(user_input,prep):
    df_in=pd.DataFrame([user_input])

    bins,labels=prep['tenure_bins']
    df_in['tenure_bin']=pd.cut(df_in['tenure'],bins=bins,labels=labels,include_lowest=True)

    df_in=df_in.drop(columns=['tenure'])

    df_in_enc=pd.get_dummies(df_in,drop_first=True)

    template_cols=prep['template_columns']
    df_in_enc=df_in_enc.reindex(columns=template_cols,fill_value=0)

    scaler=prep['scaler']
    x_scaled=scaler.transform(df_in_enc)

    nonzero=[(col,int(val)) for col,val in zip(prep['template_columns'] ,df_in_enc.iloc[0]) if val!=0]
    print(f"[DEBUG] Preprocess_input : nonzero_dummies={nonzero[:10]}")
    print(f"[DEBUG] preprocess_input: x_scaled_shape={x_scaled.shape},first10={x_scaled.flatten()[:10].tolist()}")
    
    return x_scaled

@st.cache_resource
def load_model(path='Models/Ada_boost_churn_model.pkl'):
    print(f"[DEBUG] Load Model /; Loading model From {path}")
    model=joblib.load(path)
    print(f"[DEBUG] load_model : Model Type = {type(model)}")
    return model


def main():
    st.set_page_config(
        page_title="Telecom Customer Churn Prediction",
        page_icon="📱",
        layout="wide"
    )

    st.title("📱 Telecom Customer Churn Prediction")
    st.markdown(
        "Predict whether a telecom customer is likely to churn based on customer, account, and billing information."
    )
    st.divider()

    df = load_data()
    prep = build_preprocessor(df)
    model = load_model()

    sample = prep["sample_df"]

    with st.form("input_form"):

        st.subheader("📝 Customer Details")

        user_input = {}

        with st.expander("👤 Customer Information", expanded=True):

            col1, col2 = st.columns(2)

            with col1:
                tenure = st.slider(
                    "Tenure (Months)",
                    min_value=int(sample["tenure"].min()),
                    max_value=int(sample["tenure"].max()),
                    value=12,
                )
                user_input["tenure"] = tenure

            with col2:
                if "SeniorCitizen" in sample.columns:
                    user_input["SeniorCitizen"] = st.selectbox(
                        "Senior Citizen",
                        sorted(sample["SeniorCitizen"].unique())
                    )

        st.divider()

        st.subheader("📋 Account & Billing Information")

        cols_to_ask = [
            c for c in sample.columns
            if c not in [
                "customerID",
                "Churn",
                "tenure",
                "tenure_bin",
                "SeniorCitizen"
            ]
        ]

        left, right = st.columns(2)

        for i, col in enumerate(cols_to_ask):

            target = left if i % 2 == 0 else right

            with target:

                if pd.api.types.is_numeric_dtype(sample[col]):

                    minv = float(sample[col].min())
                    maxv = float(sample[col].max())
                    default = float(sample[col].median())

                    if pd.api.types.is_integer_dtype(sample[col]):
                        minv = int(minv)
                        maxv = int(maxv)
                        default = int(default)

                    user_input[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=minv,
                        max_value=maxv,
                        value=default,
                    )

                else:

                    opts = sorted(sample[col].dropna().unique())

                    user_input[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        opts,
                    )

        submitted = st.form_submit_button(
            "🚀 Predict Churn",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        x_in = preprocessor_input(user_input, prep)

        pred_class = model.predict(x_in)[0]
        pred_proba = model.predict_proba(x_in)[0][1]

        churn_text = "Yes" if pred_class == 1 else "No"

        st.divider()
        st.subheader("📊 Prediction Result")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Prediction", churn_text)

        with c2:
            st.metric("Churn Probability", f"{pred_proba*100:.2f}%")

        st.progress(float(pred_proba))

        if pred_class == 1:
            st.error(
                f"⚠️ This customer is likely to churn.\n\nProbability: **{pred_proba*100:.2f}%**"
            )
        else:
            st.success(
                f"✅ This customer is likely to stay.\n\nProbability of churn: **{pred_proba*100:.2f}%**"
            )

        with st.expander("🔍 Debug Information"):

            st.write("### Raw Input")
            st.write(user_input)

            bins, labels = prep["tenure_bins"]

            df_in = pd.DataFrame([user_input])
            df_in["tenure_bin"] = pd.cut(
                df_in["tenure"],
                bins=bins,
                labels=labels,
                include_lowest=True,
            )

            st.write("### Input After tenure_bin")
            st.write(df_in)

            df_in_enc = pd.get_dummies(df_in.drop(columns=["tenure"]), drop_first=True)

            df_reindexed = df_in_enc.reindex(
                columns=prep["template_columns"],
                fill_value=0,
            )

            st.write("### Reindexed Features")
            st.write(df_reindexed)

            st.write("### Scaled Features (First 20)")
            st.write(x_in.flatten()[:20].tolist())

        st.success("Prediction Completed Successfully ✅")

    st.divider()

    st.info(
        "This application reproduces the same preprocessing used during model training, "
        "including tenure binning, one-hot encoding (drop_first=True), and feature scaling using StandardScaler."
    )



if __name__=='__main__':
    main()