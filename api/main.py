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
PENDING_FILE = "pending_students.csv"
STUDENT_FILE = "students.csv"
ATTENDANCE_FILE = "attendance.csv"

from api.database import (
    students_collection,
    attendance_collection,
    pending_collection
)
app = FastAPI()

# Allow React frontend
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

        # Convert to OpenCV image
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
            detector_backend="opencv"  # 🔥 important fix
        )

        if len(result) > 0 and not result[0].empty:
            df = result[0].sort_values(by="distance")
            best_match = df.iloc[0]

            identity_path = best_match["identity"]
            name = os.path.basename(os.path.dirname(identity_path))

            confidence = float(best_match["confidence"])

            return {
                "name": name,
                "confidence": confidence
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

        # ---------------- SAVE IMAGE ----------------
        pending_path = os.path.join(
            "pending_faces",
            name
        )

        os.makedirs(pending_path, exist_ok=True)

        file_path = os.path.join(
            pending_path,
            file.filename
        )

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # ---------------- CHECK DUPLICATE ----------------
        existing = pending_collection.find_one({
            "name": name
        })

        if existing:
            return {
                "message": "Student already requested"
            }

        # ---------------- SAVE TO MONGODB ----------------
        pending_collection.insert_one({
            "name": name,
            "student_id": student_id,
            "class_name": class_name
        })

        return {
            "message": "Request submitted"
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

        # ---------------- MOVE FACE FOLDER ----------------
        src = os.path.join("pending_faces", name)

        dest = os.path.join(DB_PATH, name)

        if os.path.exists(src):
            shutil.move(src, dest)

        # ---------------- GET STUDENT FROM PENDING ----------------
        student = pending_collection.find_one({
            "name": name
        })

        if not student:
            return {
                "message": "Student not found"
            }

        # ---------------- SAVE TO STUDENTS COLLECTION ----------------
        students_collection.insert_one({
            "name": student["name"],
            "student_id": student["student_id"],
            "class_name": student["class_name"]
        })

        # ---------------- REMOVE FROM PENDING ----------------
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

        # ---------------- CREATE STUDENT FOLDER ----------------
        student_path = os.path.join(DB_PATH, name)

        os.makedirs(student_path, exist_ok=True)

        # ---------------- SAVE IMAGE ----------------
        file_path = os.path.join(
            student_path,
            file.filename
        )

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # ---------------- CHECK DUPLICATE ----------------
        existing = students_collection.find_one({
            "name": name
        })

        if existing:
            return {
                "message": "Student already exists"
            }

        # ---------------- SAVE TO MONGODB ----------------
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

        # ---------------- DELETE IMAGE FOLDER ----------------
        path = os.path.join("pending_faces", name)

        if os.path.exists(path):
            shutil.rmtree(path)

        # ---------------- REMOVE FROM MONGODB ----------------
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

    # ---------------- CHECK SESSION ----------------
    if not attendance_active:

        return {
            "message": "Attendance is Closed"
        }

    try:

        print("📌 Attendance request:", name)

        today = datetime.now().strftime("%Y-%m-%d")

        time_now = datetime.now().strftime("%I:%M:%S %p")

        # ---------------- FIND STUDENT ----------------
        student = students_collection.find_one({
            "name": name
        })

        if not student:

            return {
                "message": "Student Not Found"
            }

        # ---------------- PREVENT DUPLICATE ----------------
        existing = attendance_collection.find_one({
            "name": name,
            "date": today
        })

        if existing:

            return {
                "message": "Already Marked Today"
            }

        # ---------------- SAVE ATTENDANCE ----------------
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