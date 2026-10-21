import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Horses or Humans", page_icon="🐴")

CLASS_NAMES = ["horses", "humans"]  # urutan sesuai folder dataset saat training


@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO("model/best.pt")


@st.cache_resource
def load_keras():
    import tensorflow as tf
    return tf.keras.models.load_model("model/classifier_model.h5", compile=False)


st.title("🐴 Horses or Humans 🧍")
st.write("Unggah gambar, lalu pilih model yang ingin dipakai di sidebar.")

mode = st.sidebar.selectbox("Pilih Model:", ["YOLOv8 (best.pt)", "CNN Keras (.h5)"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah")

    if mode == "CNN Keras (.h5)":
        classifier = load_keras()
        arr = np.array(img.resize((224, 224)), dtype="float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = classifier.predict(arr, verbose=0)[0]
        idx = int(np.argmax(pred))
        st.subheader(f"Hasil: {CLASS_NAMES[idx]}")
        st.write(f"Keyakinan: {pred[idx] * 100:.2f}%")
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"{name}: {pred[i] * 100:.2f}%")
            st.progress(float(pred[i]))
    else:
        yolo_model = load_yolo()
        r = yolo_model(img)[0]
        if yolo_model.task == "classify":
            st.subheader(f"Hasil: {r.names[r.probs.top1]}")
            st.write(f"Keyakinan: {float(r.probs.top1conf) * 100:.2f}%")
        else:
            st.image(r.plot()[..., ::-1], caption="Hasil Deteksi")