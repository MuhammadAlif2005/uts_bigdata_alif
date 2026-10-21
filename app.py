import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Horses or Humans", page_icon="🐴")

# Urutan sesuai folder dataset saat training
CLASS_NAMES = ["horses", "humans"]


@st.cache_resource
def load_models():
    yolo = YOLO("model/best.pt")
    clf = tf.keras.models.load_model("model/classifier_model.h5", compile=False)
    return yolo, clf


yolo_model, classifier = load_models()

st.title("🐴 Horses or Humans 🧍")
st.write("Unggah gambar, lalu pilih model yang ingin dipakai di sidebar.")

mode = st.sidebar.selectbox(
    "Pilih Model:", ["CNN Keras (.h5)", "YOLOv8 (best.pt)"]
)

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah")

    if mode == "CNN Keras (.h5)":
        img_resized = img.resize((224, 224))
        arr = np.array(img_resized, dtype="float32") / 255.0
        arr = np.expand_dims(arr, axis=0)

        pred = classifier.predict(arr)[0]
        idx = int(np.argmax(pred))

        st.subheader(f"Hasil: {CLASS_NAMES[idx]}")
        st.write(f"Keyakinan: {pred[idx] * 100:.2f}%")
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"{name}: {pred[i] * 100:.2f}%")
            st.progress(float(pred[i]))

    else:
        r = yolo_model(img)[0]

        if yolo_model.task == "classify":
            name = r.names[r.probs.top1]
            conf = float(r.probs.top1conf)
            st.subheader(f"Hasil: {name}")
            st.write(f"Keyakinan: {conf * 100:.2f}%")
        else:
            # Kalau suatu saat memakai model deteksi (ada kotak)
            plotted = r.plot()[..., ::-1]  # BGR -> RGB
            st.image(plotted, caption="Hasil Deteksi")