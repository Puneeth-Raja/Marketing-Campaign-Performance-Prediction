import os
from datetime import date

import joblib
import pandas as pd
import streamlit as st

APP_STYLE = """
<style>
    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #334155 100%);
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.25rem;
        box-shadow: 0 18px 36px rgba(15, 23, 42, 0.18);
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
        letter-spacing: -0.02em;
    }

    .hero p {
        margin: 0.5rem 0 0;
        opacity: 0.9;
        font-size: 0.98rem;
    }

    .section-card {
        background: white;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    }

    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 0.86rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stForm"] {
        background: transparent;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 999px;
        padding: 0.4rem 0.9rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }

    .stTabs [aria-selected="true"] {
        background: #0f172a !important;
        color: white !important;
    }
</style>
"""
st.markdown(APP_STYLE, unsafe_allow_html=True)

st.set_page_config(
    page_title="Marketing Campaign Prediction",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <div class="hero">
        <h1>Marketing Campaign Performance Prediction</h1>
        <p>Enter campaign inputs in a structured form to predict revenue and profitability.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

CAMPAIGN_TYPES = ["SEO", "Paid Ads", "Social Media", "Influencer"]
AUDIENCES = ["Youth", "Working Women", "Premium Shoppers", "Tier 2 City Customers"]
LANGUAGES = ["English", "Hindi", "Tamil"]
CHANNELS = ["Email", "Facebook", "Google", "Instagram", "WhatsApp", "YouTube"]
MODELLING_DATA_PATH = os.path.join("Dataset", "Data_for_modelling.xls")
DEFAULT_SHARED_INPUTS = {
    "campaign_type": CAMPAIGN_TYPES[0],
    "audience": AUDIENCES[0],
    "language": LANGUAGES[0],
    "channels": [],
    "impressions": 10000,
    "clicks": 500,
    "leads": 100,
    "conversions": 50,
    "acquisition_cost": 1000.0,
    "roi": 0.0,
    "engagement_score": 75.0,
    "campaign_date": date.today(),
}


def load_models() -> tuple:
    regression_model_path = os.path.join("model", "Gradient Boosting.pkl")
    classification_model_path = os.path.join("model", "profit_classifier.pkl")
    regression_model = joblib.load(regression_model_path)
    classification_model = joblib.load(classification_model_path)
    return regression_model, classification_model


def build_feature_frame(raw_input: dict) -> pd.DataFrame:
    input_df = pd.DataFrame(
        {
            "Campaign_Type": [raw_input["campaign_type"]],
            "Target_Audience": [raw_input["audience"]],
            "Duration": [raw_input.get("duration", 30)],
            "Language": [raw_input["language"]],
            "Customer_Segment": [raw_input["audience"]],
            "Channel_Used": [",".join(raw_input["channels"])],
            "Impressions": [raw_input["impressions"]],
            "Clicks": [raw_input["clicks"]],
            "Leads": [raw_input["leads"]],
            "Conversions": [raw_input["conversions"]],
            "Acquisition_Cost": [raw_input["acquisition_cost"]],
            "ROI": [raw_input.get("roi", 0.0)],
            "Engagement_Score": [raw_input["engagement_score"]],
            "Campaign_Date": [raw_input["campaign_date"]],
        }
    )

    input_df["Total_Acquisition_Cost"] = input_df["Conversions"] * input_df["Acquisition_Cost"]
    input_df["CTR"] = input_df["Clicks"] / (input_df["Impressions"] + 1)
    input_df["Conversion_Rate"] = input_df["Conversions"] / (input_df["Clicks"] + 1)
    input_df["CPC"] = input_df["Total_Acquisition_Cost"] / (input_df["Clicks"] + 1)
    input_df["CPL"] = input_df["Total_Acquisition_Cost"] / (input_df["Leads"] + 1)
    input_df["Campaign_Month"] = pd.to_datetime(input_df["Campaign_Date"]).dt.month
    input_df["Campaign_DayOfWeek"] = pd.to_datetime(input_df["Campaign_Date"]).dt.dayofweek

    for channel in CHANNELS:
        input_df[channel] = input_df["Channel_Used"].apply(lambda x: 1 if channel in x else 0)

    input_df = pd.get_dummies(
        input_df,
        columns=["Campaign_Type", "Target_Audience", "Language", "Customer_Segment"],
    )

    return input_df


def align_to_model_features(input_df: pd.DataFrame, model) -> pd.DataFrame:
    if not hasattr(model, "feature_names_in_"):
        return input_df

    required_features = list(model.feature_names_in_)
    for col in required_features:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[required_features]


def init_shared_input_state() -> None:
    for key, value in DEFAULT_SHARED_INPUTS.items():
        st.session_state.setdefault(f"shared_{key}", value)


def save_shared_inputs(raw_input: dict) -> None:
    for key in DEFAULT_SHARED_INPUTS:
        if key in raw_input and raw_input[key] is not None:
            st.session_state[f"shared_{key}"] = raw_input[key]


def load_shared_inputs(include_roi: bool = False) -> dict:
    shared_inputs = {key: st.session_state.get(f"shared_{key}", value) for key, value in DEFAULT_SHARED_INPUTS.items()}
    if not include_roi:
        shared_inputs["roi"] = None
    return shared_inputs


def get_shared_inputs() -> dict:
    return {
        key: st.session_state.get(f"shared_{key}", value)
        for key, value in DEFAULT_SHARED_INPUTS.items()
    }


def prefill_widget_keys_from_shared(prefix: str, include_roi: bool = False) -> None:
    shared_inputs = get_shared_inputs()
    keys_to_sync = [
        "campaign_type",
        "audience",
        "language",
        "channels",
        "impressions",
        "clicks",
        "leads",
        "conversions",
        "acquisition_cost",
        "engagement_score",
        "campaign_date",
    ]

    if include_roi:
        keys_to_sync.insert(9, "roi")

    for key in keys_to_sync:
        st.session_state[f"{prefix}_{key}"] = shared_inputs[key]




def collect_inputs(prefix: str, include_roi: bool = False, inherit_from_shared: bool = False) -> dict:
    defaults = load_shared_inputs(include_roi=include_roi) if inherit_from_shared else DEFAULT_SHARED_INPUTS

    with st.container(border=True):
        st.markdown('<div class="section-title">Campaign Setup</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Core campaign details used by the model.</div>', unsafe_allow_html=True)
        setup_col1, setup_col2 = st.columns(2)

        with setup_col1:
            campaign_type = st.selectbox("Campaign Type", CAMPAIGN_TYPES, index=CAMPAIGN_TYPES.index(defaults["campaign_type"]), key=f"{prefix}_campaign_type")
            language = st.selectbox("Language", LANGUAGES, index=LANGUAGES.index(defaults["language"]), key=f"{prefix}_language")

        with setup_col2:
            audience = st.selectbox("Audience Segment", AUDIENCES, index=AUDIENCES.index(defaults["audience"]), key=f"{prefix}_audience")
            campaign_date = st.date_input("Campaign Date", value=defaults["campaign_date"], key=f"{prefix}_campaign_date")

    with st.container(border=True):
        st.markdown('<div class="section-title">Channel Mix</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Select the channels used in the campaign.</div>', unsafe_allow_html=True)
        channels = st.multiselect("Select Channels Used", CHANNELS, default=defaults["channels"], key=f"{prefix}_channels")

    with st.container(border=True):
        st.markdown('<div class="section-title">Performance Inputs</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">The metrics most directly connected to prediction.</div>', unsafe_allow_html=True)
        perf_col1, perf_col2 = st.columns(2)
        with perf_col1:
            impressions = st.number_input("Impressions", min_value=0, value=defaults["impressions"], key=f"{prefix}_impressions")
            clicks = st.number_input("Clicks", min_value=0, value=defaults["clicks"], key=f"{prefix}_clicks")
            conversions = st.number_input("Conversions", min_value=0, value=defaults["conversions"], key=f"{prefix}_conversions")

        with perf_col2:
            leads = st.number_input("Leads", min_value=0, value=defaults["leads"], key=f"{prefix}_leads")
            acquisition_cost = st.number_input("Acquisition Cost", min_value=0.0, value=defaults["acquisition_cost"], key=f"{prefix}_acquisition_cost")
            roi = None
            if include_roi:
                roi = st.number_input("ROI", value=defaults["roi"], key=f"{prefix}_roi")
            engagement_score = st.number_input("Engagement Score", min_value=0.0, value=defaults["engagement_score"], key=f"{prefix}_engagement_score")

    return {
        "campaign_type": campaign_type,
        "audience": audience,
        "language": language,
        "channels": channels,
        "impressions": impressions,
        "clicks": clicks,
        "leads": leads,
        "conversions": conversions,
        "acquisition_cost": acquisition_cost,
        "roi": roi,
        "engagement_score": engagement_score,
        "campaign_date": campaign_date,
    }


try:
    revenue_model, profit_model = load_models()
except FileNotFoundError as exc:
    st.error(f"Model file not found: {exc}")
    st.stop()

init_shared_input_state()

regression_tab, classification_tab = st.tabs(["Regression Prediction", "Classification Profit Prediction"])


with regression_tab:
    st.subheader("Regression Inputs")
    reg_inputs = collect_inputs("reg", include_roi=True, inherit_from_shared=True)
    run_regression = st.button("Predict Revenue", key="run_regression")

    if run_regression:
        save_shared_inputs(reg_inputs)
        prefill_widget_keys_from_shared("clf", include_roi=False)
        reg_df = build_feature_frame(reg_inputs)
        reg_ready = align_to_model_features(reg_df.copy(), revenue_model)

        try:
            predicted_revenue = float(revenue_model.predict(reg_ready)[0])
            st.session_state["latest_predicted_revenue"] = predicted_revenue
            st.success("Revenue prediction completed.")
            st.metric("Predicted Revenue", f"Rs {predicted_revenue:,.2f}")
        except Exception as exc:
            st.error(f"Regression prediction failed: {exc}")


with classification_tab:
    st.subheader("Classification Inputs")
    clf_inputs = collect_inputs("clf", include_roi=False, inherit_from_shared=True)
    run_classification = st.button("Predict Profitability", key="run_classification")

    if run_classification:
        save_shared_inputs(clf_inputs)
        clf_df = build_feature_frame(clf_inputs)

        revenue_for_clf = st.session_state.get("latest_predicted_revenue")
        if revenue_for_clf is None:
            reg_df = build_feature_frame(clf_inputs)
            reg_ready = align_to_model_features(reg_df.copy(), revenue_model)
            try:
                revenue_for_clf = float(revenue_model.predict(reg_ready)[0])
                st.session_state["latest_predicted_revenue"] = revenue_for_clf
            except Exception as exc:
                st.error(f"Could not compute revenue for classification: {exc}")
                st.stop()

        clf_df["Revenue"] = revenue_for_clf
        clf_ready = align_to_model_features(clf_df.copy(), profit_model)

        try:
            profit_prediction = int(profit_model.predict(clf_ready)[0])
            st.success("Classification prediction completed.")
            if profit_prediction == 1:
                st.success("Campaign is predicted to be PROFITABLE")
            else:
                st.error("Campaign is predicted to make a LOSS")
        except Exception as exc:
            st.error(f"Classification prediction failed: {exc}")

