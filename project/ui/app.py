import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
)
st.title("🔬 Skin Lesion Classifier Showcase")
st.write(
    "A web application to demonstrate a fine-tuned ResNet-50 model. This UI communicates with a separate FastAPI backend for predictions."
)


@st.cache_data(ttl=600)
def get_test_image_data():
    """Fetches the list of test images and their data from the API."""
    try:
        response = requests.get(f"{API_URL}/test-images")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API to get test images: {e}")
        return None


def get_prediction(image_bytes):
    """Sends an image to the API's /predict endpoint and gets the result."""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed. Could not connect to the API: {e}")
        return None


# layout
st.sidebar.title("Options")
app_mode = st.sidebar.selectbox(
    "Choose an option", ["Upload your own image", "Try an image from the test set"]
)

if app_mode == "Upload your own image":
    st.header("Classify a new image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Sending to API for analysis..."):
            result = get_prediction(image_bytes)

        with col2:
            if result:
                st.success(f"**Prediction:** {result['prediction']}")
                st.write("---")
                st.write("**Certainty Scores:**")
                certainty_df = pd.DataFrame(
                    list(result["all_certainties"].items()),
                    columns=["Lesion Type", "Certainty"],
                ).sort_values(by="Certainty", ascending=False)

                st.dataframe(
                    certainty_df.style.format({"Certainty": "{:.2%}"}),
                    use_container_width=True,
                )
                st.bar_chart(certainty_df.set_index("Lesion Type"))

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

            image_url = selected_row['image_url']

            col1, col2 = st.columns(2)
            with col1:
                st.image(image_url, caption=f"Test Image: {selected_id}", use_column_width=True)
                ground_truth = selected_row['dx_full']
                st.info(f"**Ground Truth:** {ground_truth}")
                try:
                    image_response = requests.get(image_url)
                    image_response.raise_for_status()
                    image_bytes = image_response.content
                
                    with st.spinner("Sending to API for analysis..."):
                        result = get_prediction(image_bytes)

                    with col2:
                        if result:
                            prediction = result["prediction"]
                            if prediction == ground_truth:
                                st.success(f"**Prediction:** {prediction} (Correct)")
                            else:
                                st.error(f"**Prediction:** {prediction} (Incorrect)")

                            st.write("---")
                            st.write("**Certainty Scores:**")
                            certainty_df = pd.DataFrame(
                                list(result["all_certainties"].items()),
                                columns=["Lesion Type", "Certainty"],
                            ).sort_values(by="Certainty", ascending=False)

                            st.dataframe(
                                certainty_df.style.format({"Certainty": "{:.2%}"}),
                                use_container_width=True,
                            )
                            st.bar_chart(certainty_df.set_index("Lesion Type"))

                except requests.exceptions.RequestException as e:
                    st.error(f"Could not fetch image from API: {e}")