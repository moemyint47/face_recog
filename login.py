import cv2
import sqlite3
import numpy as np
from deepface import DeepFace
import tempfile
import os
import time
import pyttsx3
from scipy.spatial.distance import cosine

tts_engine = pyttsx3.init()

# Load stored embeddings from database
def load_embeddings_from_db():
    conn = sqlite3.connect("face_embeddings.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM faces")
    entries = []
    for name, blob in cursor.fetchall():
        embedding = np.frombuffer(blob, dtype=np.float32)
        entries.append((name, embedding))
    conn.close()
    return entries

# Extract embedding from frame using DeepFace
def get_embedding_from_frame(frame):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, frame)
        result = DeepFace.represent(img_path=temp_path, model_name='Facenet')[0]
        os.remove(temp_path)
        return np.array(result["embedding"], dtype=np.float32)
    except Exception as e:
        return None

# Compare embedding with DB
def recognize_user(embedding, db_entries, threshold=0.4):
    best_match = None
    best_distance = float('inf')
    for name, db_embedding in db_entries:
        distance = cosine(embedding, db_embedding)
        if distance < threshold and distance < best_distance:
            best_distance = distance
            best_match = name
    if best_match:
        confidence = 1 - best_distance  # Higher is better
        confidence = confidence * 100
        return best_match, confidence
    return None, None


# Crop center square from the frame
def get_center_square(frame, size=250):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    half = size // 2
    return frame[cy - half:cy + half, cx - half:cx + half]

# --- Main Loop ---
if __name__ == "__main__":
    db_entries = load_embeddings_from_db()
    if not db_entries:
        print("[WARN] No face data in database. Please enroll users first.")
        exit()

    cap = cv2.VideoCapture(0)
    square_size = 250
    detected_since = None
    detection_duration = 3  # seconds
    recognized_flag = False
    not_recognized_flag = False

    print("[INFO] Look into the center square for 3 seconds to log in...")
    print("[INFO] Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        top_left = (cx - square_size // 2, cy - square_size // 2)
        bottom_right = (cx + square_size // 2, cy + square_size // 2)

        # Draw center square
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Crop region of interest
        roi = get_center_square(frame, size=square_size)

        # Try to extract embedding
        embedding = get_embedding_from_frame(roi)

        if embedding is not None:
            if detected_since is None:
                detected_since = time.time()
            elif time.time() - detected_since >= detection_duration:
                name, confidence = recognize_user(embedding, db_entries)
                if name:
                    if not recognized_flag:
                        print("face recognized")
                        recognized_flag = True
                        not_recognized_flag = False
                        display_text = f"Welcome, {name} ({confidence:.2f})"
                        text_color = (0, 255, 0)
                        tts_engine.say(f"Welcome, {name}")
                        tts_engine.runAndWait()
                else:
                    if not not_recognized_flag:
                        print("not recognized")
                        not_recognized_flag = True
                        recognized_flag = False
                        tts_engine.say(f"Face not recognized")
                        tts_engine.runAndWait()
                detected_since = None  # Reset timer after recognition attempt
        else:
            detected_since = None  # Reset timer if face not detected

        # Show elapsed hold time
        elapsed = time.time() - detected_since if detected_since else 0
        if detected_since:
            timer_text = f"Hold still: {int(elapsed)}s"
            cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 200), 2)

        cv2.imshow("Face Login", frame)

        # Exit with ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
