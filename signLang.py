import cv2
import numpy as np
import joblib
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode


# --- STEP 1: Load Your Saved Model & Dictionary ---
@st.cache_resource
def load_ml_assets():
    # Loading the trained Random Forest model
    model = joblib.load("sign_language_rf_model.pkl")
    # Loading the label dictionary mapping {'ASL_A': 0, 'MSL_A': 1, ...}
    label_dict = joblib.load("label_dictionary.pkl")
    # Invert the dictionary so we can look up names by their index number
    inv_label_dict = {v: k for k, v in label_dict.items()}
    return model, inv_label_dict


try:
    rf_model, class_mapping = load_ml_assets()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info(
        "Make sure 'sign_language_rf_model.pkl' and 'label_dictionary.pkl' are in this exact folder."
    )
    model_loaded = False


# --- STEP 2: Video Processing Engine ---
class SignLanguageClassifier(VideoTransformerBase):
    def __init__(self):
        self.model = rf_model
        self.mapping = class_mapping

    def transform(self, frame):
        # 1. Convert WebRTC frame to an OpenCV BGR image
        img = frame.to_ndarray(format="bgr24")

        # Keep a copy of the original image dimensions to draw text over later
        orig_h, orig_w = img.shape[:2]

        if model_loaded:
            try:
                # 2. Preprocess frame to match training data specification:
                # Resize to (64, 64) just like the dataset load loop did
                resized_img = cv2.resize(img, (64, 64))

                # Flatten the image array from (64, 64, 3) to a 1D vector of 12288 elements
                flattened_img = resized_img.reshape(1, -1)

                # 3. Predict the class index
                pred_idx = self.model.predict(flattened_img)[0]

                # 4. Extract class label name (e.g., "MSL_A" or "ASL_B")
                label_name = self.mapping.get(pred_idx, "Unknown")

                # Clean up the display string to look nice (e.g., "MSL: A")
                display_text = label_name.replace("_", ": ")

            except Exception as e:
                display_text = "Processing Error"
        else:
            display_text = "Model Not Loaded"

        # 5. Draw the prediction text beautifully on the live output video
        # Background box for readability
        cv2.rectangle(img, (10, orig_h - 60), (350, orig_h - 10), (0, 0, 0), -1)
        # Prediction Text
        cv2.putText(
            img,
            display_text,
            (20, orig_h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return img


# --- STEP 3: Streamlit Web UI Interface ---
st.set_page_config(page_title="Sign Language Interpreter", page_icon="🤟")

st.title("🤟 Real-Time Sign Language Interpreter")
st.write(
    "This application uses your webcam to interpret Unified ASL & MSL signs using your trained Random Forest model."
)

if model_loaded:
    st.success("🤖 Random Forest model successfully loaded and ready!")

    # Run WebRTC Streamer
    webrtc_streamer(
        key="sign-language-classifier",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=SignLanguageClassifier,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

    st.info(
        "💡 **Tip:** Keep your hand centered and well-lit in the camera frame to mimic the training dataset properties."
    )
else:
    st.warning(
        "Please resolve the model loading path errors above to start the application."
    )
