> Abhinaya:
import streamlit as st
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import pygame
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("mixkit-signal-alert-771.wav")
MOUTH_AR_THRESH = 0.75
MOUTH_AR_CONSEC_FRAMES = 15
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
facemodel_path = "face.xml"
eyemodel_path = "eyes.h5"
try:
    predictor = dlib.shape_predictor(predictor_path)
    facemodel = cv2.CascadeClassifier(facemodel_path)
    eyemodel = load_model(eyemodel_path, compile=False)
except Exception as e:
    st.error(f"Error loading models: {e}")
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar
def detect_drowsiness(source):
    window = st.empty()
    vid = cv2.VideoCapture(source)
    yawning_counter = 0
    frame_counter = 0  
    if not vid.isOpened():
        st.error("Error: Could not open video source.")
        return
    while vid.isOpened():
        flag, frame = vid.read()
        if not flag:
            st.warning("Warning: No frame captured from video source.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[48:68]
            mar = mouth_aspect_ratio(mouth)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            if mar > MOUTH_AR_THRESH:
                yawning_counter += 1
                if yawning_counter >= MOUTH_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "Yawning", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    alert_sound.play()  # Play alert sound
            else:
                yawning_counter = 0
            faces = facemodel.detectMultiScale(frame)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                size = (224, 224)
                face_img = ImageOps.fit(Image.fromarray(face_img), size, Image.LANCZOS)
                face_img = (np.asarray(face_img).astype(np.float32) / 127.5) - 1
                face_img = np.expand_dims(face_img, axis=0)
                pred = eyemodel.predict(face_img)[0][0]
                if pred > 0.9:
                    path = f"data/frame_{frame_counter}.jpg"
                    cv2.imwrite(path, frame[y:y+h, x:x+w])
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    alert_sound.play()  # Play alert sound
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)    
        window.image(frame, channels="BGR")
        frame_counter += 1   
    vid.release()
st.set_page_config(page_title="DRIVER DROWSINESS DETECTION SYSTEM", page_icon="https://33.media.tumblr.com/5c79953db232e69e2f07f58b0a25c70f/tumblr_ncq090P25b1tlnptjo1_1280.gif")
st.title("DRIVER DROWSINESS DETECTION SYSTEM")
choice = st.sidebar.selectbox("My Menu", ("HOME", "URL", "CAMERA", "Feedback"))
if choice == "HOME":
    st.image("https://th.bing.com/th/id/OIP.gC1o75jJN-xJLKr8B0hiQwHaD4?rs=1&pid=ImgDetMain")
    st.write("This is a Driver Drowsiness Detection system developed using OpenCV and dlib")
elif choice == "URL":
    url = st.text_input("Enter the URL")
    btn = st.button("Start Detection")
    if btn and url:
        st.write(f"Starting detection from URL: {url}")
        detect_drowsiness(url)
elif choice == "CAMERA":
cam_options = ("None", 0, 1)
    cam = st.selectbox("Choose 0 for primary camera and 1 for secondary camera", cam_options)
    btn = st.button("Start Detection")
    if btn:
        if cam != "None":
            st.write(f"Starting detection from camera: {cam}")
            detect_drowsiness(int(cam))
        else:
            st.warning("Please select a valid camera.")
elif choice == "Feedback":
    st.markdown('<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSfKoxF-E03CCFog8ycsyNiaAkBqRh9Is-HbnhGQbfIl7_aUDw/viewform?embedded=true" width="640" height="1214" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>', unsafe_allow_html=True)
