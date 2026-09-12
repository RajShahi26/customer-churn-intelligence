
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("final_xgb_pipeline.pkl")
    threshold = joblib.load("churn_threshold.pkl")
    shap_explainer = joblib.load("shap_explainer.pkl")

    return model, threshold, shap_explainer


@st.cache_data
def load_customer_data():

    return pd.read_csv("customer_risk_predictions.csv")


model, threshold, shap_explainer = load_artifacts()
customer_data = load_customer_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_feature_name(feature):

    feature = feature.replace("num__", "")
    feature = feature.replace("cat__", "")
    feature = feature.replace("_", ": ")

    return feature


def get_model_features(customer):

    excluded_columns = [
        "customerID",
        "Churn",
        "tenure_group",
        "Churn_Probability",
        "Predicted_Churn",
        "Risk_Level"
    ]

    feature_columns = [
        col for col in customer_data.columns
        if col not in excluded_columns
    ]

    return customer[feature_columns].to_frame().T


def get_shap_explanation(customer):

    raw_features = get_model_features(customer)

    preprocessor = model.named_steps["preprocessor"]

    transformed = preprocessor.transform(raw_features)

    feature_names = preprocessor.get_feature_names_out()

    shap_values = shap_explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    explanation = pd.DataFrame({
        "Feature": [
            clean_feature_name(x)
            for x in feature_names
        ],
        "SHAP Value": shap_values
    })

    explanation["Absolute SHAP"] = (
        explanation["SHAP Value"].abs()
    )

    explanation["Impact"] = np.where(
        explanation["SHAP Value"] > 0,
        "Increases Churn Risk",
        "Decreases Churn Risk"
    )

    return explanation.sort_values(
        "Absolute SHAP",
        ascending=False
    )


def generate_recommendations(customer):

    recommendations = []

    if customer["Contract"] == "Month-to-month":

        recommendations.append({
            "Priority": "High",
            "Risk Factor": "Month-to-month contract",
            "Action":
                "Offer a suitable long-term contract incentive "
                "to encourage contract conversion."
        })

    if customer["tenure"] <= 6:

        recommendations.append({
            "Priority": "High",
            "Risk Factor": "Early customer tenure",
            "Action":
                "Apply an early-tenure retention campaign with "
                "proactive onboarding and engagement."
        })

    if customer["OnlineSecurity"] == "No":

        recommendations.append({
            "Priority": "Medium",
            "Risk Factor": "No Online Security",
            "Action":
                "Offer an Online Security package or trial "
                "to increase service value."
        })

    if customer["TechSupport"] == "No":

        recommendations.append({
            "Priority": "Medium",
            "Risk Factor": "No Technical Support",
            "Action":
                "Offer a Technical Support package or trial."
        })

    if customer["PaymentMethod"] == "Electronic check":

        recommendations.append({
            "Priority": "Medium",
            "Risk Factor": "Electronic check payment",
            "Action":
                "Encourage migration to an automatic payment "
                "method for easier recurring billing."
        })

    if customer["MonthlyCharges"] >= 80:

        recommendations.append({
            "Priority": "Medium",
            "Risk Factor": "High monthly charges",
            "Action":
                "Review pricing and service bundle and consider "
                "a targeted retention offer."
        })

    return recommendations


def risk_emoji(risk):

    return {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🟠",
        "Critical": "🔴"
    }.get(risk, "⚪")


def risk_message(risk):

    messages = {
        "Low":
            "Customer currently shows relatively low churn risk.",

        "Medium":
            "Customer may benefit from proactive engagement.",

        "High":
            "Customer should be prioritized for retention.",

        "Critical":
            "Immediate retention attention recommended."
    }

    return messages.get(risk, "")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        color: #666666;
        margin-top: 5px;
    }

    .risk-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 12px;
    }

    .recommendation-card {
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
    }

    .small-note {
        color: #777777;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explainable Customer Churn Prediction & Retention Decision Support System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# GLOBAL METRICS
# ============================================================

total_customers = len(customer_data)

high_risk_count = len(
    customer_data[
        customer_data["Risk_Level"].isin(
            ["High", "Critical"]
        )
    ]
)

critical_count = len(
    customer_data[
        customer_data["Risk_Level"] == "Critical"
    ]
)

predicted_churn = int(
    customer_data["Predicted_Churn"].sum()
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Test Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "High + Critical Risk",
        f"{high_risk_count:,}"
    )

with col3:
    st.metric(
        "Critical Risk",
        f"{critical_count:,}"
    )

with col4:
    st.metric(
        "Predicted Churn",
        f"{predicted_churn:,}"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Customer Analysis",
        "High-Risk Customers"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Model: Tuned XGBoost"
)

st.sidebar.caption(
    "Test ROC-AUC: 0.8479"
)

st.sidebar.caption(
    f"Operating Threshold: {threshold:.2f}"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📈 Risk Overview")

    risk_counts = (
        customer_data["Risk_Level"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High", "Critical"],
            fill_value=0
        )
    )

    st.bar_chart(risk_counts)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Risk Distribution")

        risk_table = pd.DataFrame({
            "Risk Level": risk_counts.index,
            "Customers": risk_counts.values,
            "Percentage": (
                risk_counts.values /
                total_customers * 100
            ).round(1)
        })

        risk_table["Percentage"] = (
            risk_table["Percentage"].astype(str) + "%"
        )

        st.dataframe(
            risk_table,
            use_container_width=True,
            hide_index=True
        )

    with col2:

        st.subheader("🤖 Model Information")

        model_info = pd.DataFrame({
            "Component": [
                "Algorithm",
                "Model Type",
                "Test ROC-AUC",
                "Operating Threshold",
                "Validation"
            ],
            "Value": [
                "XGBoost",
                "Gradient Boosting",
                "0.8479",
                f"{threshold:.2f}",
                "5-Fold Stratified CV"
            ]
        })

        st.dataframe(
            model_info,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader("🎯 Business Interpretation")

    st.info(
        "The model estimates the probability that a customer will "
        "churn. Customers are then segmented into Low, Medium, High, "
        "and Critical risk groups so retention efforts can be prioritized."
    )

    st.caption(
        "The 0.36 operating threshold was selected using out-of-fold "
        "predictions to maximize F1-score on the training data. "
        "ROC-AUC is threshold-independent."
    )


# ============================================================
# CUSTOMER ANALYSIS
# ============================================================

elif page == "Customer Analysis":

    st.header("🔍 Customer Analysis")

    customer_id = st.text_input(
        "Enter Customer ID",
        placeholder="Example: 5178-LMXOP"
    )

    if customer_id:

        customer_match = customer_data[
            customer_data["customerID"].str.upper()
            == customer_id.strip().upper()
        ]

        if customer_match.empty:

            st.error(
                "Customer ID not found. Please check the ID."
            )

        else:

            customer = customer_match.iloc[0]

            probability = float(
                customer["Churn_Probability"]
            )

            risk = customer["Risk_Level"]

            prediction = int(
                customer["Predicted_Churn"]
            )

            # ------------------------------------------------
            # RISK ASSESSMENT
            # ------------------------------------------------

            st.subheader("🎯 Churn Risk Assessment")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Churn Probability",
                    f"{probability:.2%}"
                )

            with col2:

                st.metric(
                    "Risk Level",
                    f"{risk_emoji(risk)} {risk}"
                )

            with col3:

                st.metric(
                    "Predicted Churn",
                    "Yes" if prediction == 1 else "No"
                )

            st.progress(
                min(probability, 1.0),
                text=f"Churn probability: {probability:.2%}"
            )

            st.info(
                f"{risk_emoji(risk)} {risk_message(risk)}"
            )

            st.divider()

            # ------------------------------------------------
            # CUSTOMER PROFILE
            # ------------------------------------------------

            st.subheader("👤 Customer Profile")

            profile_columns = [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges"
            ]

            profile_columns = [
                col for col in profile_columns
                if col in customer.index
            ]

            profile = customer[profile_columns]

            profile_df = pd.DataFrame({
                "Feature": profile.index,
                "Value": profile.values
            })

            st.dataframe(
                profile_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # ------------------------------------------------
            # SHAP EXPLANATION
            # ------------------------------------------------

            st.subheader(
                "🧠 Why is this customer at risk?"
            )

            try:

                explanation = get_shap_explanation(
                    customer
                )

                # Only meaningful contributors
                meaningful = explanation[
                    explanation["Absolute SHAP"] >= 0.05
                ]

                positive = meaningful[
                    meaningful["SHAP Value"] > 0
                ].sort_values(
                    "SHAP Value",
                    ascending=False
                ).head(5)

                negative = meaningful[
                    meaningful["SHAP Value"] < 0
                ].sort_values(
                    "SHAP Value",
                    ascending=True
                ).head(5)

                if not positive.empty:

                    st.markdown(
                        "**Factors increasing churn risk**"
                    )

                    positive_display = positive[
                        ["Feature", "SHAP Value", "Impact"]
                    ].copy()

                    positive_display["SHAP Value"] = (
                        positive_display["SHAP Value"]
                        .round(3)
                    )

                    st.dataframe(
                        positive_display,
                        use_container_width=True,
                        hide_index=True
                    )

                if not negative.empty:

                    st.markdown(
                        "**Factors reducing churn risk**"
                    )

                    negative_display = negative[
                        ["Feature", "SHAP Value", "Impact"]
                    ].copy()

                    negative_display["SHAP Value"] = (
                        negative_display["SHAP Value"]
                        .round(3)
                    )

                    st.dataframe(
                        negative_display,
                        use_container_width=True,
                        hide_index=True
                    )

                if meaningful.empty:

                    st.info(
                        "No feature had an absolute SHAP contribution "
                        "of at least 0.05 for this customer."
                    )

                st.caption(
                    "SHAP values explain the contribution of features "
                    "to the model output. They describe model behavior "
                    "and should not be interpreted as causal effects."
                )

            except Exception as e:

                st.error(
                    f"SHAP explanation could not be generated: {e}"
                )

            st.divider()

            # ------------------------------------------------
            # RETENTION RECOMMENDATIONS
            # ------------------------------------------------

            st.subheader(
                "💡 Retention Recommendations"
            )

            recommendations = generate_recommendations(
                customer
            )

            if recommendations:

                for rec in recommendations:

                    priority = rec["Priority"]

                    if priority == "High":
                        icon = "🔴"
                    else:
                        icon = "🟡"

                    st.markdown(
                        f"""
                        <div class="recommendation-card">
                        <b>{icon} {priority} Priority</b><br><br>
                        <b>Risk Factor:</b> {rec["Risk Factor"]}<br>
                        <b>Recommended Action:</b> {rec["Action"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.success(
                    "No specific rule-based retention "
                    "recommendations were triggered."
                )

            st.caption(
                "Recommendations are business-rule heuristics "
                "based on customer characteristics and risk factors. "
                "They are decision-support suggestions, not guaranteed outcomes."
            )


# ============================================================
# HIGH-RISK CUSTOMERS
# ============================================================

elif page == "High-Risk Customers":

    st.header("🚨 High-Risk Customers")

    high_risk = customer_data[
        customer_data["Risk_Level"].isin(
            ["High", "Critical"]
        )
    ].copy()

    high_risk = high_risk.sort_values(
        "Churn_Probability",
        ascending=False
    )

    st.write(
        f"Showing **{len(high_risk):,}** customers "
        "classified as High or Critical risk."
    )

    # ------------------------------------------------
    # FILTER
    # ------------------------------------------------

    selected_risk = st.multiselect(
        "Filter by Risk Level",
        ["High", "Critical"],
        default=["High", "Critical"]
    )

    filtered = high_risk[
        high_risk["Risk_Level"].isin(
            selected_risk
        )
    ]

    # ------------------------------------------------
    # SUMMARY
    # ------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Customers in View",
            f"{len(filtered):,}"
        )

    with col2:

        if len(filtered) > 0:

            st.metric(
                "Average Churn Probability",
                f"{filtered['Churn_Probability'].mean():.2%}"
            )

    with col3:

        if len(filtered) > 0:

            st.metric(
                "Highest Churn Probability",
                f"{filtered['Churn_Probability'].max():.2%}"
            )

    st.divider()

    # ------------------------------------------------
    # TABLE
    # ------------------------------------------------

    display_columns = [
        "customerID",
        "Churn_Probability",
        "Risk_Level",
        "Predicted_Churn",
        "tenure",
        "Contract",
        "InternetService",
        "MonthlyCharges",
        "PaymentMethod"
    ]

    display_columns = [
        col for col in display_columns
        if col in filtered.columns
    ]

    display_df = filtered[display_columns].copy()

    display_df["Churn_Probability"] = (
        display_df["Churn_Probability"]
        .map(lambda x: f"{x:.2%}")
    )

    display_df["Predicted_Churn"] = (
        display_df["Predicted_Churn"]
        .map({
            0: "No",
            1: "Yes"
        })
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------------
    # DOWNLOAD
    # ------------------------------------------------

    download_df = filtered[display_columns].copy()

    csv = download_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download High-Risk Customer List",
        data=csv,
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )

    st.caption(
        "Customers are ranked by predicted churn probability. "
        "This list is intended to support prioritization of retention efforts."
    )
