import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


@st.cache_data
def load_data(path='data/Customer-Churn.csv'):
    df=pd.read_csv(path)
    return df

@st.cache_resource
def load_model(path="models/ada_boost_churn_model.pkl"):
    print(f"[DEBUG] load model : Loading model from {path}")
    model=joblib.load(path)
    print(f"[DEBUG] load_model : model type={type(model)}")
    return model

@st.cache_resource
def build_preprocessor(df):
    telco=df.copy()
    telco['TotalCharges']=pd.to_numeric(telco['TotalCharges'],errors='coerce')
    telco.dropna(how='any',inplace=True)

    bins=[0,12,24,36,48,60,72]
    labels=['1-12','13-24','25-36','37-48','49-60','61-72']
    telco['tenure_bin']=pd.cut(telco['tenure'],bins=bins,labels=labels,include_lowest=True)

    x_template=telco.drop(columns=['customerID','Churn','tenure'])
    x_template=pd.get_dummies(x_template,drop_first=True)

    scaler=StandardScaler()
    scaler.fit(x_template)

    print(f"[DEBUG] build_preprocessor: template_cols= {x_template.shape[1]}, template_rows={x_template.shape[0]}")

    return {
        'template_cols':list(x_template.columns),
        'scaler':scaler,
        'tenure_bins':(bins,labels),
        'sample_df':telco
    }



def main():
    st.set_page_config(page_title='Churn Prediction',layout='centered')
    st.title("Telecom Customer Churn Prediction ⚡")

    df=load_data()
    prep=build_preprocessor(df)
    model=load_model()




if __name__=='__main__':
    main()