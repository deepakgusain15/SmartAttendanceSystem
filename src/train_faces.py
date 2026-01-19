import face_recognition
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(BASE_DIR, "dataset")
encodings_path = os.path.join(BASE_DIR, "encodings", "encodings.pkl")

known_encodings = []
known_names = []

print("Training started...")

for person in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person)
    if not os.path.isdir(person_dir):
        continue

    print("Processing:", person)

    for img in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img)

        image = face_recognition.load_image_file(img_path)
        locations = face_recognition.face_locations(image)

        if len(locations) == 0:
            continue

        encoding = face_recognition.face_encodings(image, locations)[0]
        known_encodings.append(encoding)
        known_names.append(person)

data = {"encodings": known_encodings, "names": known_names}

with open(encodings_path, "wb") as f:
    pickle.dump(data, f)

print("Training complete ")
print("Encodings saved in encodings/encodings.pkl")

