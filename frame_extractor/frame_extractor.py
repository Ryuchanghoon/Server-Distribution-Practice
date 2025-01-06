from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import mediapipe as mp
from collections import Counter
import requests
import os

app = FastAPI()

templates = Jinja2Templates(directory="templates")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def calculate_vertical_distance_cm(landmark1, landmark2, frame_height, distance_to_camera_cm=60, camera_fov_degrees=25):
    if landmark1 is None or landmark2 is None:
        return None
    landmark1_pixel = landmark1[1] * frame_height
    landmark2_pixel = landmark2[1] * frame_height
    pixel_distance = np.abs(landmark1_pixel - landmark2_pixel)
    real_height_cm = 2 * distance_to_camera_cm * np.tan(np.radians(camera_fov_degrees / 2))
    cm_per_pixel = real_height_cm / frame_height
    return pixel_distance * cm_per_pixel

def calculate_angle(p1, p2):
    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    if angle < 0:
        angle += 360
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

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_video(video_url: str = Form(...), interval: int = Form(...)):
    try:
        response = requests.get(video_url, stream=True)
        if response.status_code != 200:
            return JSONResponse(content={"error": "Failed to download video"}, status_code=400)

        file_location = "Save_video.mp4"
        with open(file_location, "wb") as buffer:
            buffer.write(response.content)

        cap = cv2.VideoCapture(file_location)
        frameRate = cap.get(5)
        interval = 5  # 5초 간격 추출 고정.
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
                    left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    left_ear = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y]

                    vertical_distance_cm = calculate_vertical_distance_cm(left_shoulder, left_ear, frame.shape[0])
                    angle = calculate_angle(left_ear, left_shoulder)
                    adjusted_angle = adjust_angle(angle)
                    angle_status = evaluate_angle_condition(adjusted_angle)

                    landmarks_info.append({
                        "left_shoulder": left_shoulder,
                        "left_ear": left_ear,
                        "vertical_distance_cm": vertical_distance_cm,
                        "adjusted_angle": adjusted_angle
                    })
                    angle_conditions.append(angle_status)

        status_frequencies = Counter(angle_conditions)
        cap.release()
        os.remove(file_location)

        return JSONResponse(content={
            "landmarks_info": landmarks_info,
            "status_frequencies": dict(status_frequencies)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)