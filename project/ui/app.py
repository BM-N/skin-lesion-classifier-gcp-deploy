import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
)

if "log_df" not in st.session_state:
    st.session_state["log_df"] = pd.DataFrame()

st.title("🔬 Skin Lesion Classifier Showcase")
st.write(
    "A web application to demonstrate a fine-tuned ResNet-50 model inference, deployment, and live monitoring."
    "This UI communicates with a separate FastAPI backend for predictions."
)


@st.cache_data(ttl=300)
def get_monitoring_data():
    """Fetches the full prediction log from the API."""
    if not API_URL:
        return pd.DataFrame()
    try:
        response = requests.get(f"{API_URL}/monitoring-data")
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        print(f"Could not get monitoring data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_test_image_data():
    """Fetches the list of test images and their data from the API."""
    if not API_URL:
        st.error("API_URL environment variable is not set.")
        return None
    try:
        response = requests.get(f"{API_URL}/test-images")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API to get test images: {e}")
        return None


def get_prediction_from_upload(image_bytes):
    if not API_URL:
        return None
    try:
        predict_url = f"{API_URL}/predict"
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(predict_url, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed. Could not connect to the API: {e}")
        return None


def get_prediction_from_url(image_url: str):
    if not API_URL:
        return None
    try:
        predict_url = f"{API_URL}/predict-from-url"
        payload = {"image_url": image_url}
        response = requests.post(predict_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed: {e}")
        return None


def display_prediction_and_update_log(result, ground_truth=None):
    """
    Displays the prediction result and triggers a refresh of the monitoring data.
    """
    if not result:
        st.error("Prediction failed. No result received from API.")
        return

    if result.get("error"):
        st.warning(f"**{result.get('prediction', 'Info')}**: {result.get('error')}")
    else:
        if ground_truth:
            if result["prediction"] == ground_truth:
                st.success(f"**Prediction:** {result['prediction']} (Correct)")
            else:
                st.error(f"**Prediction:** {result['prediction']} (Incorrect)")
        else:
            st.success(f"**Prediction:** {result['prediction']}")

    st.write("---")

    if result.get("all_certainties"):
        st.write("**Certainty Scores:**")
        certainty_df = pd.DataFrame(
            list(result["all_certainties"].items()),
            columns=["Lesion Type", "Certainty"],
        ).sort_values(by="Certainty", ascending=False)

        st.dataframe(
            certainty_df.style.format({"Certainty": "{:.2%}"}), use_container_width=True
        )
        st.bar_chart(certainty_df.set_index("Lesion Type"))

    st.info("Updating monitoring dashboard...")
    st.cache_data.clear()
    st.rerun()


def render_monitoring_dashboard(log_df):
    """Takes a dataframe of logs and displays the monitoring charts."""
    st.header("📊 Live Monitoring Dashboard")
    st.write("This dashboard updates with every new prediction made above.")

    if log_df.empty:
        st.info(
            "No prediction data has been logged yet. Make a prediction to see the dashboard."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Predictions Logged", len(log_df))
    with col2:
        ood_count = len(log_df[log_df["predicted_class"] == "Out-of-Distribution"])
        st.metric("Out-of-Distribution Inputs Detected", ood_count)

    st.subheader("Prediction Confidence Distribution")
    fig = px.histogram(
        log_df.dropna(subset=["confidence_score"]),
        x="confidence_score",
        nbins=20,
        title="Model Confidence",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Drift Monitoring")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("OOD Score Distribution")
        fig_ood = px.histogram(
            log_df, x="ood_distance_score", nbins=20, title="Out-of-Distribution Score"
        )
        st.plotly_chart(fig_ood, use_container_width=True)
    with col2:
        st.subheader("Image Brightness Distribution")
        fig_bright = px.histogram(
            log_df.dropna(subset=["average_brightness"]),
            x="average_brightness",
            nbins=20,
            title="Average Brightness of Inputs",
        )
        st.plotly_chart(fig_bright, use_container_width=True)


# layout
st.sidebar.title("Options")
app_mode = st.sidebar.selectbox(
    "Choose an option",
    ["Try an image from the test set", "Upload your own image"],
    key="app_mode",
    index=None,
    placeholder="Select an inference mode...",
)

prediction_result_placeholder = st.empty()

if app_mode == "Upload your own image":
    st.header("Classify a new image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        with prediction_result_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    image_bytes, caption="Uploaded Image", use_container_width=True
                )

            with st.spinner("Sending to API for analysis..."):
                result = get_prediction_from_upload(image_bytes)

            with col2:
                display_prediction_and_update_log(result)

else:
    st.header("Select an image from the official test set")
    test_image_data = get_test_image_data()

    if test_image_data:
        df = pd.DataFrame(test_image_data)
        df["display_option"] = df["image_id"] + " (" + df["dx_full"] + ")"

        selected_option = st.selectbox(
            "Select a test image:",
            options=df["display_option"],
            index=None,
            placeholder="Choose an image to test...",
        )

        if selected_option:
            selected_id = selected_option.split(" ")[0]
            selected_row = df[df["image_id"] == selected_id].iloc[0]

            image_url = selected_row["image_url"]
            ground_truth = selected_row["dx_full"]

            with prediction_result_placeholder.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        image_url,
                        caption=f"Test Image: {selected_id}",
                        use_container_width=True,
                    )
                    st.info(f"**Ground Truth:** {ground_truth}")

                with st.spinner("Analyzing the image..."):
                    result = get_prediction_from_url(image_url)
                with col2:
                    display_prediction_and_update_log(result, ground_truth)
st.divider()

if "log_df" not in st.session_state or st.session_state.log_df.empty:
    st.session_state.log_df = get_monitoring_data()

render_monitoring_dashboard(st.session_state.log_df)
