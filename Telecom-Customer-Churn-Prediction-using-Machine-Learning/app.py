import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import streamlit as st

@st.cache_data
def load_data(path='/data/Customer-Churn.csv'):
    df=pd.read_csv(path)
    return df

@st.cache_resource
def preprocessor_builder(df):
    telco=df.copy()
    telco['TotalCharges']=pd.to_numeric(telco['TotalCharges'],errors='coerce')
    telco.dropna(how='any',inplace=True)

    bins=[0,12,24,36,48,60,72]
    labels=['1-12','13-24','25-36','37-48','49-60','61-72']
    telco['tenure_bin']=pd.cut(telco['tenure'],bins=bins,labels=labels,include_lowest=True)

    x_template=telco.drop(columns=['customerID','tenure','Churn'])
    x_template=pd.get_dummies(x_template,drop_first=True)

    scaler=StandardScaler()
    scaler.fit(x_template)

def main():
    st.set_page_config(page_title='Churn Prediction',layout='centered')
    st.title("Telecom Customer Churn Prediction ⚡")