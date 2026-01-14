import asyncio
import time
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import joblib
from fastapi import FastAPI, WebSocket

# ================= WINDOWS FIX =================
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ================= APP =================
app = FastAPI()

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "best_model.pkl")
encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

# ================= MEDIAPIPE =================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= CONSTANTS =================
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
COORDS = 3

FEATURES_PER_FRAME = (POSE_LANDMARKS + 2 * HAND_LANDMARKS) * COORDS
WINDOW_SIZE = 30
INFERENCE_INTERVAL = 4.0  # seconds

# ================= STATE =================
feature_buffer = deque(maxlen=WINDOW_SIZE)
collecting = False
window_start_time = None

# ================= LANDMARK EXTRACTION =================
def extract_landmarks(results):
    features = []

    def extract(block, count):
        if block:
            for lm in block.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * count)

    extract(results.pose_landmarks, POSE_LANDMARKS * COORDS)
    extract(results.left_hand_landmarks, HAND_LANDMARKS * COORDS)
    extract(results.right_hand_landmarks, HAND_LANDMARKS * COORDS)

    return np.array(features, dtype=np.float32)

# ================= WEBSOCKET =================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    global collecting, window_start_time

    while True:
        message = await ws.receive()

        # ================= CONTROL (TEXT) =================
        if message["type"] == "websocket.receive" and "text" in message:
            text = message["text"]

            if text == "START":
                collecting = True
                feature_buffer.clear()
                window_start_time = time.time()
                await ws.send_text("Timer started")
                continue

            if text == "STOP":
                collecting = False
                feature_buffer.clear()
                window_start_time = None
                await ws.send_text("Timer stopped")
                continue

        # ================= IGNORE FRAMES IF NOT COLLECTING =================
        if not collecting:
            continue

        # ================= FRAME (BYTES) =================
        if "bytes" not in message:
            continue

        data = message["bytes"]

        np_frame = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        if not (results.left_hand_landmarks or results.right_hand_landmarks):
            continue

        feature_buffer.append(extract_landmarks(results))

        # ================= 4-SECOND TIMER =================
        elapsed = time.time() - window_start_time

        if elapsed >= INFERENCE_INTERVAL:
            X = np.array(feature_buffer)

            if len(X) >= WINDOW_SIZE:
                idx = np.linspace(0, len(X) - 1, WINDOW_SIZE).astype(int)
                X = X[idx]
            else:
                padding = np.zeros((WINDOW_SIZE - len(X), FEATURES_PER_FRAME))
                X = np.vstack([X, padding])

            X = X.flatten().reshape(1, -1)

            pred = model.predict(X)[0]
            label = encoder.inverse_transform([pred])[0]

            await ws.send_text(label)

            # Reset for next 4-second window
            feature_buffer.clear()
            window_start_time = time.time()
