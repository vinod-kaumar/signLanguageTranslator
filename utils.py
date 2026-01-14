import cv2
import mediapipe as mp
import numpy as np

# ================= CONSTANTS =================
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
COORDS = 3

FEATURES_PER_FRAME = (POSE_LANDMARKS + 2 * HAND_LANDMARKS) * COORDS
MAX_FRAMES = 30
FINAL_FEATURE_SIZE = FEATURES_PER_FRAME * MAX_FRAMES

mp_holistic = mp.solutions.holistic

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

# ================= VIDEO PROCESSING =================
def process_video(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            frames.append(extract_landmarks(results))

    cap.release()

    if len(frames) == 0:
        return np.zeros(FINAL_FEATURE_SIZE)

    frames = np.array(frames)

    if len(frames) >= max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = frames[idx]
    else:
        padding = np.zeros((max_frames - len(frames), FEATURES_PER_FRAME))
        frames = np.vstack([frames, padding])

    return frames.flatten()
if __name__ == "__main__":
    pass