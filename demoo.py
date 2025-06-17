import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import csv
import smtplib
from email.message import EmailMessage
import pyttsx3

# Email configuration
EMAIL_ADDRESS = "kalaniharshita0202@gmail.com"  # Sender email
EMAIL_PASSWORD = "nlgh wpcp kaom boty"    # App password or email password
EMAIL_RECEIVER = "harshitakalani0303@gmail.com"  # Receiver email

# Create necessary folders
os.makedirs("data", exist_ok=True)
os.makedirs("unknown_faces", exist_ok=True)

def speak_alert(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

def generate_dataset(id):
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
            return cropped_face

    cap = cv2.VideoCapture(0)
    img_id = 0
    print(f"Starting dataset generation for user {id}. Press Enter to stop or collect 200 samples.")

    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"data/user.{id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped Face", face)

        if cv2.waitKey(1) == 13 or img_id == 200:  # Enter key or 200 images
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Dataset Generation Completed...")
#generate_dataset()
def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')  # grayscale
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    print("Model Training Completed...")
#train_classifier("data")
def send_email_alert(image_path):
    try:
        msg = EmailMessage()
        msg["Subject"] = "Unknown Person Detected"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = EMAIL_RECEIVER
        msg.set_content("An unknown person was detected by the face recognition system.")

        with open(image_path, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(image_path)

        msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print(f"Email sent to {EMAIL_RECEIVER} with attachment {file_name}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def recognize_face():
    def log_recognition(user_id):
        with open("log.csv", "a", newline="") as file:
            writer = csv.writer(file)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([user_id, now])
            print(f"Logged: User {user_id} at {now}")

    def save_unknown_face(face_img):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unknown_faces/unknown_{now}.jpg"
        cv2.imwrite(filename, face_img)
        print(f"Unknown face saved: {filename}")
        speak_alert("Unknown person detected")
        send_email_alert(filename)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.exists("classifier.xml"):
        print("Model not found. Please train the classifier first.")
        return

    clf.read("classifier.xml")

    cap = cv2.VideoCapture(0)
    print("Starting face recognition. Press Enter to stop.")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id, confidence = clf.predict(roi_gray)

            if confidence < 50:
                label = f"User {id}"
                log_recognition(id)
            else:
                label = "Unknown"
                face_img = frame[y:y + h, x:x + w]
                save_unknown_face(face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == 13:  # Enter key to stop
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stopped.")
recognize_face()