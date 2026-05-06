from pymongo import MongoClient

# ---------------- MONGODB CONNECTION ----------------

MONGO_URL = "mongodb+srv://dssingla:deepak12@jan2025.6vqml.mongodb.net/?retryWrites=true&w=majority&appName=jan2025"

client = MongoClient(MONGO_URL)

# Database
mongo_db = client["attendance_system"]

# Collections
students_collection = mongo_db["students"]

attendance_collection = mongo_db["attendance"]

pending_collection = mongo_db["pending_students"]

print("✅ MongoDB Connected")