from fastapi import FastAPI, WebSocket
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

app = FastAPI()

# =======================
# Load ML artifacts
# =======================
model = joblib.load("models/best_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# =======================
# MediaPipe
# =======================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

WINDOW_SIZE = 30
feature_buffer = deque(maxlen=WINDOW_SIZE)
prediction_buffer = deque(maxlen=10)

def extract_landmarks(results):
    features = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0]*33*3)

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0]*21*3)

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0]*21*3)

    return np.array(features, dtype=np.float32)

# =======================
# WebSocket endpoint
# =======================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    while True:
        data = await ws.receive_bytes()
        np_frame = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        if not results.left_hand_landmarks and not results.right_hand_landmarks:
            await ws.send_text("No sign detected")
            continue

        features = extract_landmarks(results)
        feature_buffer.append(features)

        if len(feature_buffer) == WINDOW_SIZE:
            X = np.array(feature_buffer).flatten().reshape(1, -1)
            pred = model.predict(X)[0]
            prediction_buffer.append(pred)

            final_pred = Counter(prediction_buffer).most_common(1)[0][0]
            label = encoder.inverse_transform([final_pred])[0]

            await ws.send_text(label)