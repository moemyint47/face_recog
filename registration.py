import cv2
import sqlite3
import numpy as np
from deepface import DeepFace
import tempfile
import os

# Connect to SQLite DB
conn = sqlite3.connect("face_embeddings.db")
cursor = conn.cursor()

# Create table for embeddings
cursor.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    embedding BLOB
)
""")
conn.commit()

def save_embedding_to_db(name, embedding):
    embedding_blob = sqlite3.Binary(np.array(embedding).astype(np.float32).tobytes())
    cursor.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, embedding_blob))
    conn.commit()
    print(f"[INFO] Embedding for '{name}' saved in database.")

def capture_photo_from_webcam():
    cap = cv2.VideoCapture(0)
    print("[INFO] Press SPACE to capture image or ESC to exit.")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Webcam - Press SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == 32:  # SPACE
            cap.release()
            cv2.destroyAllWindows()
            return frame

def extract_face_embedding_from_image(image_path, name="Unknown"):
    try:
        result = DeepFace.represent(img_path=image_path, model_name='Facenet')[0]
        embedding = result["embedding"]
        save_embedding_to_db(name, embedding)
    except Exception as e:
        print(f"[ERROR] Face detection/embedding failed: {e}")

def extract_face_embedding_from_frame(frame, name="Unknown"):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
            temp_path = tmpfile.name
            cv2.imwrite(temp_path, frame)

        result = DeepFace.represent(img_path=temp_path, model_name='Facenet')[0]
        os.remove(temp_path)
        embedding = result["embedding"]
        save_embedding_to_db(name, embedding)
    except Exception as e:
        print(f"[ERROR] Face detection/embedding failed: {e}")

# ---- Main flow ----
if __name__ == "__main__":
    print("Choose input method:")
    print("1. Capture photo from webcam")
    print("2. Use local image file path")
    choice = input("Enter 1 or 2: ")

    name = input("Enter name for the face: ")

    if choice == "1":
        frame = capture_photo_from_webcam()
        if frame is not None:
            extract_face_embedding_from_frame(frame, name)
    elif choice == "2":
        image_path = input("Enter full path to image file: ")
        if os.path.exists(image_path):
            extract_face_embedding_from_image(image_path, name)
        else:
            print("[ERROR] File does not exist.")
    else:
        print("[ERROR] Invalid choice.")

    conn.close()
