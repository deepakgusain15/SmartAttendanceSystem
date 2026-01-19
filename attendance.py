import cv2
import face_recognition
import pickle
import csv
import os
import numpy as np  
from datetime import datetime
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encodings_path = os.path.join(BASE_DIR, "encodings", "encodings.pkl")
attendance_file = os.path.join(BASE_DIR, "attendance", "attendance.csv")

if not os.path.exists(encodings_path):
    print("Error: encodings.pkl nahi mili! Pehle train_faces.py chala.")
    exit()

with open(encodings_path, "rb") as f:
    data = pickle.load(f)

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

marked = set()

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("System Ready... Looking for face.")

face_locations = []
face_encodings = []
face_names = []
frame_count = 0

attendance_marked_success = False 

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 5 == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = [] 

        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            
            best_match_index = np.argmin(face_distances)
            
            name = "Unknown"

            if face_distances[best_match_index] < 0.50:
                name = data["names"][best_match_index]

            face_names.append(name)

            if name != "Unknown":
                if name not in marked:
                    marked.add(name)
                    now = datetime.now()
                    
                    with open(attendance_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
                    
                    print(f"Success! {name} marked present.")
                    attendance_marked_success = True  

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        display_text = name + " (Marked)" if attendance_marked_success else name
        cv2.putText(frame, display_text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("Smart Attendance", frame)

    if attendance_marked_success:
        cv2.waitKey(2000) 
        break 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
print("System Closed.")