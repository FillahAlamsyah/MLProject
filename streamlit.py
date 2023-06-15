import cv2
import streamlit as st
import numpy as np


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


def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("Aplikasi ini memungkinkan Anda untuk memanipulasi filter gambar dan melakukan face recognition menggunakan OpenCV dan Streamlit")

    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')
    apply_face_recognition = st.sidebar.checkbox('Face Recognition')

    video_capture = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while True:
        _, frame = video_capture.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        processed_image = brighten_image(frame_rgb, brightness_amount)

        if apply_enhancement_filter:
            processed_image = enhance_details(processed_image)

        if apply_face_recognition:
            faces = detect_faces(frame)
            processed_image = draw_face_boxes(processed_image, faces)

        frame_placeholder.image(processed_image)

    video_capture.release()


if __name__ == '__main__':
    main_loop()

'''from mtcnn.mtcnn import MTCNN
import streamlit as st
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

choice = st.selectbox("Select Option",[
    "Face Detection",
    "Face Detection 2",
    "Face Verification"
])

def main():
    fig = plt.figure()
    if choice == "Face Detection":
        uploaded_file = st.file_uploader("Choose File", type=["jpg","png"])
        if uploaded_file is not None:
            data = asarray(Image.open(uploaded_file))
            plt.axis("off")
            plt.imshow(data)
            ax = plt.gca()
          
            detector = MTCNN()
            faces = detector.detect_faces(data)
            for face in faces:
                x, y, width, height = face['box']
                rect = Rectangle((x, y), width, height, fill=False, color='maroon')
                ax.add_patch(rect)
                for _, value in face['keypoints'].items():
                    dot = Circle(value, radius=2, color='maroon')
                    ax.add_patch(dot)
            st.pyplot(fig)
            
            results = detector.detect_faces(pixels)
            x1, y1, width, height = results[0]["box"]
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
    
    elif choice == "Face Detection 2":
            uploaded_file = st.file_uploader("Choose File", type=["jpg","png"])
            if uploaded_file is not None:
                column1, column2 = st.beta_columns(2)
                image = Image.open(uploaded_file)
                with column1:
                    size = 450, 450
                    resized_image = image.thumbnail(size)
                    image.save("thumb.png")
                    st.image("thumb.png")
                pixels = asarray(image)
                plt.axis("off")
                plt.imshow(pixels)
                detector = MTCNN()
                results = detector.detect_faces(pixels)
                x1, y1, width, height = results[0]["box"]
                x2, y2 = x1 + width, y1 + height
                face = pixels[y1:y2, x1:x2]
                image = Image.fromarray(face)
                image = image.resize((224, 224)) 
                face_array = asarray(image)
                with column2:
                     plt.imshow(face_array)
                     st.pyplot(fig)

    def extract_face(file):
        pixels = asarray(file)
        plt.axis("off")
        plt.imshow(pixels)
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        x1, y1, width, height = results[0]["box"]
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize((224, 224))
        face_array = asarray(image)
        return face_array

if __name__ == "__main__":
    main()
'''
