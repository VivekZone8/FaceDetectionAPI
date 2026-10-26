
import json
import os

import cv2
import numpy as np
import pyodbc

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from deepface import DeepFace

app = FastAPI(title="Face Registration and Verification API")

# Update this server name according to your SSMS connection.

CONNECTION_STRING = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    r"SERVER=localhost\SQLEXPRESS;"
    "DATABASE=FaceRecognitionDB;"
    "Trusted_Connection=yes;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)


MODEL_NAME = "Facenet512"
MAX_IMAGE_SIZE = 10 * 1024 * 1024


def get_connection():
    return pyodbc.connect(CONNECTION_STRING)


def get_embedding(image_bytes):
    if not image_bytes or len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Image must be between 1 byte and 10 MB"
        )

    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    try:
        results = DeepFace.represent(
            img_path=image,
            model_name=MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=True
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not detect a usable face. Try a clear, front-facing image."
        ) from exc

    if len(results) != 1:
        raise HTTPException(
            status_code=400,
            detail="Image must contain exactly one detectable face"
        )

    return np.array(results[0]["embedding"], dtype=np.float32)


def cosine_distance(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 1.0

    return float(1 - np.dot(a, b) / denominator)


@app.get("/")
def home():
    return {"message": "Face Recognition API is running"}


@app.post("/register-face")
async def register_face(
    user_id: int = Form(...),
    file: UploadFile = File(...)
):
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid user ID")

    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Upload JPG, PNG or WEBP")

    image_bytes = await file.read()
    embedding = get_embedding(image_bytes)

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT UserId FROM FaceUsers WHERE UserId = ?",
                user_id
            )

            if cursor.fetchone():
                raise HTTPException(
                    status_code=409,
                    detail="This user is already registered"
                )

            cursor.execute(
                """
                INSERT INTO FaceUsers (UserId, FaceEmbedding)
                VALUES (?, ?)
                """,
                user_id,
                json.dumps(embedding.tolist())
            )

            conn.commit()

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Database operation failed. Check SQL connection settings."
        ) from exc

    return {
        "success": True,
        "message": "Face registered successfully",
        "user_id": user_id
    }


@app.post("/verify-face")
async def verify_face(
    user_id: int = Form(...),
    file: UploadFile = File(...)
):
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid user ID")

    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Upload JPG, PNG or WEBP")

    image_bytes = await file.read()
    new_embedding = get_embedding(image_bytes)

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT FaceEmbedding FROM FaceUsers WHERE UserId = ?",
                user_id
            )

            row = cursor.fetchone()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Database operation failed. Check SQL connection settings."
        ) from exc

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="User is not registered"
        )

    stored_embedding = np.array(
        json.loads(row[0]),
        dtype=np.float32
    )

    distance = cosine_distance(new_embedding, stored_embedding)

    # Initial threshold only; calibrate using consented test images.
    matched = distance < 0.30

    return {
        "success": True,
        "user_id": user_id,
        "verified": matched,
        "distance": round(distance, 4),
        "message": (
            "Face matched"
            if matched
            else "Face did not match"
        )
    }
