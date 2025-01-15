from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
import numpy as np
import cv2
import aiohttp
import tempfile
import os

app = FastAPI()


try:
    model = tf.keras.models.load_model('final_baro_model.h5')
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

async def download_video(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    content = await response.read()
                    tmp_file.write(content)
                    return tmp_file.name
    return None

def extract_frames(video_file, interval=5):
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    images = []

    if not cap.isOpened():
        raise RuntimeError("Failed to open video file.")

    while cap.isOpened():
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        if int(frame_id) % int(frame_rate * interval) == 0:
            resized_frame = cv2.resize(frame, (28, 28))  # 모델 input 데이터 28 by 28
            normalized_frame = resized_frame / 255.0
            images.append(normalized_frame)

    cap.release()
    return np.array(images)

def process_model_predictions(images):
    predictions_proba = model.predict(images)
    result = np.argmax(predictions_proba, axis=1)
    scores = (np.max(predictions_proba, axis=1) * 100).tolist()

    hunched_posture_label = 0
    normal_posture_label = 1

    total_predictions = len(result)
    hunched_count = np.sum(result == hunched_posture_label)
    normal_count = np.sum(result == normal_posture_label)

    hunched_ratio = (hunched_count / total_predictions) * 100 if total_predictions > 0 else 0
    normal_ratio = (normal_count / total_predictions) * 100 if total_predictions > 0 else 0

    return {
        "predictions": result.tolist(),
        "scores": scores,
        "posture_ratios": {
            "hunched_ratio": hunched_ratio,
            "normal_ratio": normal_ratio
        }
    }

@app.post("/predict")
async def predict(video_url: str = Form(...)):
    try:
        video_file = await download_video(video_url)
        if video_file is None:
            raise HTTPException(status_code=400, detail="Video download failed")

        images = extract_frames(video_file, interval=5)

        if len(images) == 0:
            raise HTTPException(status_code=400, detail="No frames extracted from the video")

        prediction_results = process_model_predictions(images)

        os.remove(video_file)

        return JSONResponse(content={
            "status": "success",
            "data": prediction_results
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=400)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)