import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- STEP 1: Initialize MediaPipe Hands & Drawing Utilities ---
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- STEP 1: Safe Initialization for Modern MediaPipe ---
try:
    # Modern approach / Fallback for newer MediaPipe versions
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
except AttributeError:
    # Legacy approach
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

# Load the hands tracking module safely
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


# --- STEP 2: Pure Geometric Rule Engine (No .pkl file needed!) ---
def recognize_gesture(landmarks):
    """
    Decodes the hand skeleton joints using geometric logic for alphabets.
    'landmarks' contains 21 points with normalized x, y, z coordinates.
    """
    # 1. Extract raw coordinates
    wrist = landmarks[0]

    # Thumb joints
    thumb_cmc = landmarks[1]
    thumb_mcp = landmarks[2]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]

    # Finger tips
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Finger Knuckles (MCP joints)
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    # Finger PIP joints (middle knuckles)
    index_pip = landmarks[6]
    middle_pip = landmarks[10]
    ring_pip = landmarks[14]
    pinky_pip = landmarks[18]

    # 2. Calculate Boolean States (Is the finger straight up?)
    # Remember: y decreases as a joint moves UP the screen
    index_up = index_tip.y < index_pip.y and index_pip.y < index_mcp.y
    middle_up = middle_tip.y < middle_pip.y and middle_pip.y < middle_mcp.y
    ring_up = ring_tip.y < ring_pip.y and ring_pip.y < ring_mcp.y
    pinky_up = pinky_tip.y < pinky_pip.y and pinky_pip.y < pinky_mcp.y

    # All fingers curled tightly
    all_fist = not index_up and not middle_up and not ring_up and not pinky_up

    # 3. Alphabet Decision Tree

    # --- GROUP 1: All Fingers Extended ---
    if index_up and middle_up and ring_up and pinky_up:
        # B: Hand flat, thumb folded over palm (inside the index position horizontally)
        if thumb_tip.x > index_mcp.x:
            return "Letter: B"
        return "Letter: Full Open Palm"

    # --- GROUP 2: Straight Vector Extended Single/Double Fingers ---
    # D: Index pointing straight up, others curled down in a ring with thumb
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "Letter: D"

    # F: Index and Thumb touching down, Middle, Ring, and Pinky straight up
    if not index_up and middle_up and ring_up and pinky_up:
        return "Letter: F"

    # I: Only pinky finger extended straight up
    if pinky_up and not index_up and not middle_up and not ring_up:
        return "Letter: I"

    # L: Index up, Thumb extended widely out to the side horizontally
    if index_up and not middle_up and not ring_up and not pinky_up:
        if abs(thumb_tip.x - index_mcp.x) > 0.1:
            return "Letter: L"

    # W: Index, Middle, and Ring up. Pinky curled down
    if index_up and middle_up and ring_up and not pinky_up:
        return "Letter: W"

    # --- GROUP 3: Multiple Extended Fingers V, U, K, Y ---
    if index_up and middle_up and not ring_up and not pinky_up:
        # Check distance between Index and Middle tips to distinguish U and V
        finger_spread = abs(index_tip.x - middle_tip.x)
        # K: Thumb pointing upwards touching the middle knuckle line
        if thumb_tip.y < middle_mcp.y:
            return "Letter: K"
        elif finger_spread > 0.06:
            return "Letter: V"
        else:
            return "Letter: U"

    # Y: Thumb and Pinky stretched completely outwards, middle three curled
    if pinky_up and not index_up and not middle_up and not ring_up:
        if abs(thumb_tip.x - wrist.x) > 0.12:
            return "Letter: Y"

    # --- GROUP 4: Fist / Closed Variants (A, C, E, O, S, T, M, N) ---
    if all_fist:
        # A: Thumb tucked straight up along the side of the index finger
        if thumb_tip.y < index_pip.y and thumb_tip.x < index_mcp.x:
            return "Letter: A"
        # E: All fingertips curling down tightly, resting right on top of the thumb
        if thumb_tip.y > ring_pip.y:
            return "Letter: E"
        # S: Thumb crossing straight across the middle front of the fist fingers
        if (
            thumb_tip.x > index_pip.x
            and thumb_tip.x < ring_pip.x
            and thumb_tip.y < middle_pip.y
        ):
            return "Letter: S"
        # C / O: Checking for an open curved space/profile width
        if abs(index_mcp.x - thumb_tip.x) > 0.08:
            return "Letter: C / O"

    # --- GROUP 5: Horizontal / Pointing Forwards (G, H, P, Q) ---
    # In these signs, fingers point sideways (x changes instead of y)
    index_pointing_sideways = abs(index_tip.x - index_mcp.x) > 0.12
    middle_pointing_sideways = abs(middle_tip.x - middle_mcp.x) > 0.12

    if index_pointing_sideways and not pinky_up and not ring_up:
        if middle_pointing_sideways:
            return "Letter: H"
        else:
            return "Letter: G"

    # --- GROUP 6: Specialized Complex Sign Overlays ---
    # Rock / Spider-Man structure used as placeholder for specific assignments
    if index_up and pinky_up and not middle_up and not ring_up:
        return "Special: I Love You Sign"

    return "Tracking Hand (Unknown / Transitioning)"


# --- STEP 3: Video Processing Worker ---
class MediaPipeVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert WebRTC frame to standard OpenCV BGR image
        img = frame.to_ndarray(format="bgr24")
        orig_h, orig_w = img.shape[:2]

        # MediaPipe requires RGB color format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        display_text = "No hand detected"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the smooth skeletal lines directly onto the webcam image
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Rule-base decoding of the 21 coordinate joints
                display_text = recognize_gesture(hand_landmarks.landmark)

        # Draw UI overlay bounding bar
        cv2.rectangle(img, (10, orig_h - 60), (450, orig_h - 10), (0, 0, 0), -1)
        cv2.putText(
            img,
            display_text,
            (20, orig_h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return img


# --- STEP 4: Streamlit Interface Layout ---
st.set_page_config(page_title="MediaPipe Sign Interpreter", page_icon="hand")

st.title("Real-Time MediaPipe Gesture Interpreter")
st.write(
    "This version tracks your hand's physical joints to interpret gestures instantly without relying on a static pixel model."
)

st.info(
    "Why this is better: It tracks the coordinates of your joints, so your room background or lighting won't mess up the prediction accuracy!"
)

# Launch WebRTC Streamer
webrtc_streamer(
    key="mediapipe-sign-language",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=MediaPipeVideoTransformer,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},  # Fallback server
        ]
    },
    media_stream_constraints={"video": True, "audio": False},
)
