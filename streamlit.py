import os
os.system("pip install -r requirements.txt")
from mtcnn.mtcnn import MTCNN
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
