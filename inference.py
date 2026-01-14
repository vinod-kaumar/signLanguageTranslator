import joblib
from utils import process_video

MODEL_PATH = "models/best_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

def predict_video(video_path):
    features = process_video(video_path)
    pred = model.predict([features])[0]
    return encoder.inverse_transform([pred])[0]

# ================= TEST =================
if __name__ == "__main__":
    video_path = "aman.mp4" 
    print("Prediction:", predict_video(video_path))
