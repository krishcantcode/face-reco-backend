from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from deepface import DeepFace
import os
import csv
from fastapi import File, UploadFile, Form
from datetime import datetime
import shutil
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET")
)

from api.database import (
    students_collection,
    attendance_collection,
    pending_collection
)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "registered_faces"

@app.get("/")
def home():
    return {"message": "Backend running"}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Convert bytes → OpenCV image
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"name": "Error", "confidence": 0}

        print("Image received:", img.shape)

        result = DeepFace.find(
            img_path=img,
            db_path=DB_PATH,
            model_name="VGG-Face",
            enforce_detection=False,
            detector_backend="opencv"
        )

        if len(result) > 0 and not result[0].empty:

            df = result[0].sort_values(by="distance")
            best_match = df.iloc[0]

            identity_path = best_match["identity"]
            name = os.path.basename(os.path.dirname(identity_path))

            distance = float(best_match["distance"])

            confidence = round((1 - distance) * 100, 2)

            if distance < 0.7:

                return {
                    "name": name,
                    "confidence": confidence,
                    "distance": distance
                }

            else:

                return {
                    "name": "Unknown",
                    "confidence": confidence,
                    "distance": distance
                }

        return {"name": "Unknown", "confidence": 0}

    except Exception as e:
        print("ERROR:", e)
        return {"name": "Error", "confidence": 0}

from fastapi import Form

@app.post("/request-student")
async def request_student(
    name: str = Form(...),
    student_id: str = Form(...),
    class_name: str = Form(...),
    file: UploadFile = File(...)
):
    try:

        existing = pending_collection.find_one({
            "name": name
        })

        if existing:
            return {
                "message": "Student already requested"
            }

        pending_path = os.path.join(
            "pending_faces",
            name
        )

        os.makedirs(pending_path, exist_ok=True)

        file_path = os.path.join(
            pending_path,
            file.filename
        )

        contents = await file.read()

        with open(file_path, "wb") as f:
            f.write(contents)

        upload_result = cloudinary.uploader.upload(
            file_path,
            folder="pending_faces",
            public_id=name,
            overwrite=True
        )

        image_url = upload_result["secure_url"]

        pending_collection.insert_one({
            "name": name,
            "student_id": student_id,
            "class_name": class_name,
            "image_url": image_url,
            "local_path": file_path
        })

        return {
            "message": "Request submitted",
            "image_url": image_url
        }

    except Exception as e:

        print("ERROR:", e)

        return {
            "message": str(e)
        }
    
@app.get("/pending-students")
def get_pending():

    data = []

    try:

        students = pending_collection.find()

        for row in students:

            data.append({
                "Name": row.get("name"),
                "ID": row.get("student_id"),
                "Class": row.get("class_name")
            })

        return data

    except Exception as e:

        print("ERROR:", e)

        return []



@app.post("/approve-student")
async def approve_student(name: str = Form(...)):
    try:

        src = os.path.join("pending_faces", name)

        dest = os.path.join(DB_PATH, name)

        if os.path.exists(src):
            shutil.move(src, dest)

        student = pending_collection.find_one({
            "name": name
        })

        if not student:
            return {
                "message": "Student not found"
            }

        students_collection.insert_one({
            "name": student["name"],
            "student_id": student["student_id"],
            "class_name": student["class_name"]
        })

        pending_collection.delete_one({
            "name": name
        })

        return {
            "message": "Approved Successfully"
        }

    except Exception as e:

        print("ERROR:", e)

        return {
            "message": str(e)
        }
    

@app.post("/add-student")
async def add_student(
    name: str = Form(...),
    student_id: str = Form(...),
    class_name: str = Form(...),
    file: UploadFile = File(...)
):
    try:

        print("Received:", name, student_id, class_name)

        student_path = os.path.join(DB_PATH, name)

        os.makedirs(student_path, exist_ok=True)

        file_path = os.path.join(
            student_path,
            file.filename
        )

        with open(file_path, "wb") as f:
            f.write(await file.read())

        existing = students_collection.find_one({
            "name": name
        })

        if existing:
            return {
                "message": "Student already exists"
            }

        students_collection.insert_one({
            "name": name,
            "student_id": student_id,
            "class_name": class_name
        })

        return {
            "message": "Student added successfully"
        }

    except Exception as e:

        print("ERROR:", e)

        return {
            "message": str(e)
        }
    
@app.post("/reject-student")
async def reject_student(name: str = Form(...)):
    try:

        path = os.path.join("pending_faces", name)

        if os.path.exists(path):
            shutil.rmtree(path)

        pending_collection.delete_one({
            "name": name
        })

        return {
            "message": "Rejected Successfully"
        }

    except Exception as e:

        print("ERROR:", e)

        return {
            "message": str(e)
        }
    
@app.post("/mark-attendance")
async def mark_attendance(name: str = Form(...)):

    global attendance_active

    if not attendance_active:

        return {
            "message": "Attendance is Closed"
        }

    try:

        print("📌 Attendance request:", name)

        today = datetime.now().strftime("%Y-%m-%d")

        time_now = datetime.now().strftime("%I:%M:%S %p")

        student = students_collection.find_one({
            "name": name
        })

        if not student:

            return {
                "message": "Student Not Found"
            }

        existing = attendance_collection.find_one({
            "name": name,
            "date": today
        })

        if existing:

            return {
                "message": "Already Marked Today"
            }

        attendance_collection.insert_one({

            "name": student["name"],

            "student_id": student["student_id"],

            "class_name": student["class_name"],

            "date": today,

            "time": time_now,

            "status": "Present"
        })

        return {
            "message": "Attendance Marked"
        }

    except Exception as e:

        print("❌ ERROR:", e)

        return {
            "message": str(e)
        }
        
@app.get("/attendance")
def get_attendance():

    data = []

    try:

        records = attendance_collection.find()

        for row in records:

            data.append({

                "Name": row.get("name"),

                "ID": row.get("student_id"),

                "Class": row.get("class_name"),

                "Date": row.get("date"),

                "Time": row.get("time"),

                "Status": row.get("status")

            })

        return data

    except Exception as e:

        print("ERROR:", e)

        return []
    

attendance_active = False
@app.post("/start-attendance")
def start_attendance():
    global attendance_active

    attendance_active = True

    return {
        "message": "Attendance Started"
    }

@app.post("/stop-attendance")
def stop_attendance():
    global attendance_active

    attendance_active = False

    return {
        "message": "Attendance Stopped"
    }

@app.get("/students")
def get_students():

    data = []

    try:

        students = students_collection.find()

        for row in students:

            data.append({

                "Name": row.get("name"),

                "ID": row.get("student_id"),

                "Class": row.get("class_name")

            })

        return data

    except Exception as e:

        print("ERROR:", e)

        return []