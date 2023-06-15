import cv2
import streamlit as st
import numpy as np
import pickle


def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def draw_face_boxes(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image


def save_database(database, filename):
    with open(filename, 'wb') as file:
        pickle.dump(database, file)


def load_database(filename):
    try:
        with open(filename, 'rb') as file:
            database = pickle.load(file)
            return database
    except FileNotFoundError:
        return {}


def main_loop():
    st.title("Security System Unknown People Detection Web App")
    st.subheader("Based Face Recognition")

    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')
    apply_face_recognition = st.sidebar.checkbox('Face Recognition')

    available_cameras = {
        'Camera 0': 0,
        'Camera 1': 1,
        'Camera 2': 'http://192.168.183.6:8501'
    }

    selected_camera = st.sidebar.selectbox("Pilih Kamera", list(available_cameras.keys()))

    video_capture = cv2.VideoCapture(available_cameras[selected_camera])
    frame_placeholder = st.empty()
    face_label_text = st.empty()

    while True:
        _, frame = video_capture.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        processed_image = brighten_image(frame_rgb, brightness_amount)

        if apply_enhancement_filter:
            processed_image = enhance_details(processed_image)

        if apply_face_recognition:
            faces = detect_faces(frame)
            processed_image = draw_face_boxes(processed_image, faces)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_image = frame_rgb[y:y+h, x:x+w]
                face_label = f"Face 1"

                if face_label not in database:
                    database[face_label] = face_image

                save_database(database, database_file)

                face_label_text.text(face_label)

        frame_placeholder.image(processed_image, channels="RGB")

    video_capture.release()


if __name__ == '__main__':
    database_file = "database.pkl"
    database = load_database(database_file)
    main_loop()
