import streamlit as st
import pickle
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import shap


def add_sidebar():
    st.sidebar.header("Enter Transaction Details")
    type = st.sidebar.selectbox("Select Transaction Type", ["PAYMENT","TRANSFER","CASH_OUT","DEPOSIT","CASH_IN"])
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, step=10.0)
    oldbalanceOrg = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0, value=0.0, step= 10.0)
    newbalanceOrig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0, value=0.0, step= 10.0)
    oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, step= 10.0)
    newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, step= 10.0)
    input = pd.DataFrame([{
            "type": type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
            }])
    return input


def main():
    st.set_page_config(
        page_title="Smart Fraud Detector",
        page_icon= "🏛️",
        layout = "wide",
        initial_sidebar_state="expanded" 
    )

    model = pickle.load(open("src/model.pkl", "rb"))
    explainer = pickle.load(open("src/explainer.pkl", "rb"))

    input_data = add_sidebar()

    with st.container():
        st.title("Smart Fraud Detector - ML Risk Analysis")
        st.write("This app helps in detecting potential fraudulent transactions using a machine learning model. "
            "It analyzes transaction details and predicts whether an activity is likely to be **fraudulent** or **legitimate**. "
            "You can connect this tool to your financial systems for real-time monitoring, or test it by entering sample transaction data. "
            "The goal is to assist in preventing financial losses and ensuring safer digital transactions.")

    col1, col2 = st.columns([4,1])

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0] 

        with col1:

            st.subheader(f"Prediction:")

            if prediction == 1:
                st.error("This transaction can be a fraud")
            else:
                st.success("This transaction looks like safe")
        
        with col2:
            st.metric(label="Fraud Probability", value=f"{prob[1]*100:.2f}%")
        
        with col1:
            
            preprocessor = model.named_steps["prep"]

            # Step 2: Transform the input (raw dataframe) -> numeric features
            X_transformed = preprocessor.transform(input_data)
            feature_names = model.named_steps['prep'].get_feature_names_out()

            # Step 3: Run SHAP on the transformed data and the logistic regression
            shap_values = explainer(X_transformed)

            st.markdown("### *Why did the model predict this?*")

            
            st.subheader("Summary Plot")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
            st.pyplot(fig)

            st.subheader("Waterfall Plot")
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

            st.subheader("Feature Importance")
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values, show=False)
            st.pyplot(fig)


            


if __name__ == '__main__':
    main()