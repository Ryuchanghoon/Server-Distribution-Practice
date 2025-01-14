from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import mediapipe as mp
import cv2
import numpy as np
import math
from collections import Counter
import uvicorn
import aiohttp
import tempfile
import os


app = FastAPI()


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

templates = Jinja2Templates(directory="templates")


def calculate_angle(p1, p2):
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    if angle < 0:
        angle += 360
    if angle > 180:
        angle = 360 - angle
    return angle


def adjust_angle(angle):
    if angle > 180:
        angle = 360 - angle
    return angle


def evaluate_angle_condition(angle):
    adjusted_angle = adjust_angle(angle)

    if 165 <= adjusted_angle <= 180:
        return 'Fine'
    elif 150 <= adjusted_angle < 165:
        return 'Danger'
    elif 135 <= adjusted_angle < 150:
        return 'Serious'
    elif adjusted_angle < 135:
        return 'Very Serious'


def calculate_horizontal_distance(landmark1, landmark2, frame_width, distance_to_camera_cm=60, camera_fov_degrees=25):
    if landmark1 is None or landmark2 is None:
        return None

    landmark1_pixel_x = landmark1[0] * frame_width
    landmark2_pixel_x = landmark2[0] * frame_width

    pixel_distance_x = np.abs(landmark1_pixel_x - landmark2_pixel_x)

    real_width_cm = 2 * distance_to_camera_cm * np.tan(np.radians(camera_fov_degrees / 2))

    cm_per_pixel = real_width_cm / frame_width
    horizontal_distance_cm = pixel_distance_x * cm_per_pixel

    return horizontal_distance_cm


def extract_pose_data(video_file, interval=5):
    cap = cv2.VideoCapture(video_file)
    frameRate = cap.get(5) 
    landmarks_info = []
    angle_conditions = []

    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % (frameRate * interval) == 0:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                frame_width = frame.shape[1]
                left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                left_ear = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y]

                horizontal_distance_cm = calculate_horizontal_distance(left_shoulder, left_ear, frame_width)
                angle = calculate_angle(left_ear, left_shoulder)
                adjusted_angle = adjust_angle(angle)
                angle_status = evaluate_angle_condition(adjusted_angle)

                landmarks_info.append((left_shoulder, left_ear, adjusted_angle, horizontal_distance_cm))
                angle_conditions.append(angle_status)

    status_frequencies = Counter(angle_conditions)
    cap.release()
    return landmarks_info, dict(status_frequencies)


async def download_video(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(await response.read())
                    return tmp_file.name
    return None


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(video_url: str = Form(...)):
    try:
        video_file = await download_video(video_url)
        if video_file is None:
            raise HTTPException(status_code=400, detail="Video download failed")

        landmarks_info, status_frequencies = extract_pose_data(video_file)
        os.remove(video_file)

        return JSONResponse(content={
            'landmarks_info': [
                {
                    'left_shoulder': {'x': info[0][0], 'y': info[0][1]},
                    'left_ear': {'x': info[1][0], 'y': info[1][1]},
                    'angle': info[2],
                    'horizontal_distance_cm': info[3]
                } for info in landmarks_info
            ],
            'status_frequencies': status_frequencies
        })
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=400)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)