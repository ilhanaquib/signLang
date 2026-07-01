import cv2
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- STEP 1: Initialize CVZone Hand Detector ---
# detectionCon=0.7 matches our previous confidence threshold
detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.7)

# --- STEP 2: Geometric Alphabet Engine ---
def recognize_gesture(lmList):
    """
    Decodes the hand skeleton joints using cvzone landmarks.
    lmList contains 21 coordinates where each point is [x, y, z].
    In cvzone, coordinates are pixel values (not normalized 0-1 values), 
    so 'y' still decreases as you move UP the screen.
    """
    # Extract raw coordinate arrays
    wrist = lmList[0]
    
    thumb_tip = lmList[4]
    index_tip = lmList[8]
    middle_tip = lmList[12]
    ring_tip = lmList[16]
    pinky_tip = lmList[20]
    
    index_mcp = lmList[5]
    middle_mcp = lmList[9]
    ring_mcp = lmList[13]
    pinky_mcp = lmList[17]
    
    index_pip = lmList[6]
    middle_pip = lmList[10]
    ring_pip = lmList[14]
    pinky_pip = lmList[18]

    # Calculate Boolean States (Is the finger straight up?)
    index_up = index_tip[1] < index_pip[1] and index_pip[1] < index_mcp[1]
    middle_up = middle_tip[1] < middle_pip[1] and middle_pip[1] < middle_mcp[1]
    ring_up = ring_tip[1] < ring_pip[1] and ring_pip[1] < ring_mcp[1]
    pinky_up = pinky_tip[1] < pinky_pip[1] and pinky_pip[1] < pinky_mcp[1]
    
    all_fist = not index_up and not middle_up and not ring_up and not pinky_up

    # Alphabet Decision Tree
    # --- B: Flat Hand ---
    if index_up and middle_up and ring_up and pinky_up:
        if thumb_tip[0] > index_mcp[0]: 
            return "Letter: B"
        return "Letter: Full Open Palm"

    # --- D: Index Up ---
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "Letter: D"
        
    # --- F: OK Sign Layout ---
    if not index_up and middle_up and ring_up and pinky_up:
        return "Letter: F"

    # --- I: Pinky Up ---
    if pinky_up and not index_up and not middle_up and not ring_up:
        return "Letter: I"

    # --- L: L Shape ---
    if index_up and not middle_up and not ring_up and not pinky_up:
        if abs(thumb_tip[0] - index_mcp[0]) > 40: # Pixel distance check
            return "Letter: L"

    # --- W: Three Fingers ---
    if index_up and middle_up and ring_up and not pinky_up:
        return "Letter: W"

    # --- U, V, K ---
    if index_up and middle_up and not ring_up and not pinky_up:
        finger_spread = abs(index_tip[0] - middle_tip[0])
        if thumb_tip[1] < middle_mcp[1]:
            return "Letter: K"
        elif finger_spread > 30:
            return "Letter: V"
        else:
            return "Letter: U"

    # --- Y ---
    if pinky_up and not index_up and not middle_up and not ring_up:
        if abs(thumb_tip[0] - wrist[0]) > 60:
            return "Letter: Y"

    # --- Fist Variants (A, E, S, C/O) ---
    if all_fist:
        if thumb_tip[1] < index_pip[1] and thumb_tip[0] < index_mcp[0]:
            return "Letter: A"
        if thumb_tip[1] > ring_pip[1]:
            return "Letter: E"
        if thumb_tip[0] > index_pip[0] and thumb_tip[0] < ring_pip[0] and thumb_tip[1] < middle_pip[1]:
            return "Letter: S"
        if abs(index_mcp[0] - thumb_tip[0]) > 40:
            return "Letter: C / O"

    # --- Sideways Signs (G, H) ---
    index_pointing_sideways = abs(index_tip[0] - index_mcp[0]) > 50
    middle_pointing_sideways = abs(middle_tip[0] - middle_mcp[0]) > 50

    if index_pointing_sideways and not pinky_up and not ring_up:
        if middle_pointing_sideways:
            return "Letter: H"
        else:
            return "Letter: G"

    return "Tracking Hand"

# --- STEP 3: Video Processing Worker ---
class CVZoneVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if img is None:
            return frame
            
        orig_h, orig_w = img.shape[:2]

        # cvzone finds hands and draws the skeleton lines automatically
        hands, img = detector.findHands(img, draw=True, flipType=False)
        display_text = "No hand detected"

        if hands:
            hand1 = hands[0]
            lmList = hand1["lmList"]  # List of 21 Landmark points
            display_text = recognize_gesture(lmList)

        # Draw UI text layout overlay
        cv2.rectangle(img, (10, orig_h - 60), (450, orig_h - 10), (0, 0, 0), -1)
        cv2.putText(img, display_text, (20, orig_h - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
        return img

# --- STEP 4: Streamlit Layout ---
st.set_page_config(page_title="Sign Interpreter", page_icon="hand")
st.title("Real-Time Gesture Interpreter")
st.write("Tracks joint coordinates to decode alphabets seamlessly.")

webrtc_streamer(
    key="cvzone-sign-language",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=CVZoneVideoTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)