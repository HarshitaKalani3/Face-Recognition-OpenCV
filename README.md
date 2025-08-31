# Real-Time Face Recognition System using OpenCV - Face Recognizer
This project is a Python-based Face Recognition System using OpenCV. It recognizes by LBPH recognizer the known faces and sends an email alert with a photo if an unknown person is detected.
-----

## Folders Created Automatically
-'data'- It stores images of known users for training.
-'unknown_faces'- It saves photos of unknown person.

## Requirements
Make sure one install's Python libraries given below: 
'''bash
pip install opencv-python opencv-contrib-python pillow numpy
'''
## How To Use
### 1. Collect Face Data
To collect face data(images data) of a user:
'''python
generate_dataset(id)
'''
- It captures 200 face images and stores them in the 'data' folder.
### 2. Train the Model
After collecting images, train the system:
'''python
trainclassifier("data")
'''
- This will create a trained model file 'classifier.xml'.
### 3. Run Face Recognition
'''python to handle images
recognize_face()
'''
- If a known person is recognized, it shows ID and logs time in the log.csv file.
- If an unknown face is detected:
    - Saves their photo in 'unknown_faces/'
    - Sends an email alert with the unknown face image attached.

## Email Alert Setup
Add below lines in the code with real email info:
'''python
EMAIL_ADDRESS = "personemail@gmail.com"
EMAIL_PASSWORD = "password"
EMAIL_RECEIVER = "receiveremail@gmail.com"
'''

## Logging
- All known users are logged in 'log.csv' file with date and time.

## Tools Used
- OpenCV for face detection.
- LBPH (Local Binary Pattern Histogram) model for face recognition.
- Pillow (PIL) to handle images.
- SMTP (Simple Mail Transfer Protocol) for sending emails.

## Example Input
'''python 
generate_dataset(1)
generate_dataset(2)
train_classifier("data")
recognize_face()
'''
## Example Output
'''
Logged: User 2 at 2025-06-17 19:45:22
Unknown face saved:
unknown_faces/unknown_20250617_195102.jpg
Email sent to harshitakalani0303@gmail.com with attachment unknown_20250617_195102.jpg
'''

## Under Guidance of: Mr. Nitin Patil Sir
## Author: Harshita Kalani (https://github.com/HarshitaKalani3)
