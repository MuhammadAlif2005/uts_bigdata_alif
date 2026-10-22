import hashlib
import html
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).parent

# ==========================================================
# Konfigurasi halaman
# ==========================================================
st.set_page_config(
    page_title="Horses or Humans AI",
    page_icon="🐴",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = ["horses", "humans"]  # urutan sesuai folder dataset saat training
EMOJI = {"horses": "🐴", "humans": "🧍"}
SAMPLE_DIR = BASE_DIR / "sample_image"
IMG_TYPES = {".jpg", ".jpeg", ".png"}


def load_css():
    """Membaca style.css lalu menyuntikkannya ke halaman."""
    css_path = BASE_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


load_css()

st.markdown(
    """
    <div class="hero">
        <h1>🐴 Horses or Humans 🧍</h1>
        <p>Klasifikasi gambar dengan CNN Keras dan YOLOv8. Unggah foto, ambil dari kamera, atau pilih contoh.</p>
    </div>
    <div class="steps">
        <div class="step"><b>1</b>Pilih model di sidebar</div>
        <div class="step"><b>2</b>Unggah atau pilih gambar</div>
        <div class="step"><b>3</b>Lihat hasil dan riwayat</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# Pemuatan model (lazy: hanya dimuat saat dibutuhkan)
# PyTorch/Ultralytics selalu dimuat SEBELUM TensorFlow.
# ==========================================================
@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def load_yolo():
    from ultralytics import YOLO

    return YOLO(str(BASE_DIR / "model" / "best.pt"))


@st.cache_resource(show_spinner="Memuat model CNN Keras...")
def load_keras():
    import tensorflow as tf

    return tf.keras.models.load_model(str(BASE_DIR / "model" / "classifier_model.h5"), compile=False)


def predict_keras(img: Image.Image) -> dict:
    model = load_keras()
    arr = np.array(img.resize((224, 224)), dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    t0 = time.perf_counter()
    probs = model.predict(arr, verbose=0)[0]
    ms = (time.perf_counter() - t0) * 1000
    return {"name": "CNN Keras", "labels": CLASS_NAMES, "probs": np.asarray(probs), "ms": ms}


def predict_yolo(img: Image.Image) -> dict:
    model = load_yolo()
    t0 = time.perf_counter()
    r = model(img, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    if model.task != "classify":
        st.error("best.pt adalah model deteksi, bukan klasifikasi. Gunakan model YOLOv8-cls.")
        st.stop()
    labels = [r.names[i] for i in range(len(r.names))]
    probs = r.probs.data.cpu().numpy()
    return {"name": "YOLOv8", "labels": labels, "probs": probs, "ms": ms}


# ==========================================================
# Komponen tampilan
# ==========================================================
def render_result(res: dict, threshold: float):
    probs, labels = res["probs"], res["labels"]
    idx = int(np.argmax(probs))
    label, conf = labels[idx], float(probs[idx])
    low = conf < threshold

    bars = ""
    for i, name in enumerate(labels):
        pct = float(probs[i]) * 100
        top = "" if i == idx else "rendah"
        bars += (
            f'<div class="bar-row"><div class="bar-head">'
            f'<span>{EMOJI.get(name, "")} {html.escape(str(name))}</span><span>{pct:.1f}%</span></div>'
            f'<div class="bar-bg"><div class="bar-fill {top}" style="width:{pct:.1f}%"></div></div></div>'
        )

    st.markdown(
        f"""
        <div class="card {'low' if low else ''}">
            <div class="model">{html.escape(res['name'].upper())}</div>
            <div class="emoji">{EMOJI.get(label, '❓')}</div>
            <div class="label">{html.escape(str(label))}</div>
            {bars}
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Keyakinan", f"{conf * 100:.1f}%")
    c2.metric("Waktu", f"{res['ms']:.0f} ms")
    if low:
        st.warning(f"Keyakinan di bawah ambang {threshold * 100:.0f}%. Hasil kurang bisa diandalkan.")
    return label, conf


def add_history(source: str, res: dict, label: str, conf: float):
    st.session_state.history.insert(
        0,
        {
            "Waktu": datetime.now().strftime("%H:%M:%S"),
            "Sumber": source,
            "Model": res["name"],
            "Hasil": label,
            "Keyakinan (%)": round(conf * 100, 2),
            "Waktu (ms)": round(res["ms"]),
        },
    )


# ==========================================================
# Sidebar
# ==========================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_key" not in st.session_state:
    st.session_state.last_key = None

with st.sidebar:
    st.header("⚙️ Pengaturan")
    mode = st.radio("Model", ["YOLOv8 (best.pt)", "CNN Keras (.h5)", "Bandingkan keduanya"])
    threshold = st.slider("Ambang keyakinan", 0.5, 0.99, 0.80, 0.01)
    source_type = st.radio("Sumber gambar", ["📤 Unggah", "📷 Kamera", "🖼️ Contoh"])
    st.divider()
    st.caption(
        "Model hanya mengenal dua kelas: horses dan humans. "
        "Gambar lain tetap akan dipaksa masuk ke salah satunya."
    )

# ==========================================================
# Pengambilan gambar
# ==========================================================
img, source_name = None, ""

if source_type == "📤 Unggah":
    up = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if up is not None:
        img, source_name = Image.open(up).convert("RGB"), up.name
elif source_type == "📷 Kamera":
    cam = st.camera_input("Ambil foto")
    if cam is not None:
        img, source_name = Image.open(cam).convert("RGB"), "kamera"
else:
    samples = (
        sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in IMG_TYPES)
        if SAMPLE_DIR.exists()
        else []
    )
    if not samples:
        st.info("Folder sample_image kosong atau tidak ditemukan. Tambahkan beberapa gambar ke folder tersebut.")
    else:
        choice = st.selectbox("Pilih gambar contoh", [p.name for p in samples])
        img, source_name = Image.open(SAMPLE_DIR / choice).convert("RGB"), choice

# ==========================================================
# Prediksi
# ==========================================================
tab_pred, tab_hist = st.tabs(["🔍 Prediksi", "🕑 Riwayat"])

with tab_pred:
    if img is None:
        st.markdown(
            '<div class="kosong"><div class="ikon">🖼️</div>'
            "<h3>Belum ada gambar</h3>"
            "<p>Pilih sumber gambar di sidebar, lalu unggah atau pilih gambar untuk memulai.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        left, right = st.columns([1, 1.4], gap="large")
        with left:
            st.image(img, caption=source_name, width="stretch")

        key = (hashlib.md5(img.tobytes()).hexdigest(), mode)
        is_new = key != st.session_state.last_key

        with right:
            with st.spinner("Menganalisis gambar..."):
                if mode == "Bandingkan keduanya":
                    results = [predict_yolo(img), predict_keras(img)]  # YOLO dulu, baru TensorFlow
                elif mode == "YOLOv8 (best.pt)":
                    results = [predict_yolo(img)]
                else:
                    results = [predict_keras(img)]

            cols = st.columns(len(results))
            outputs = []
            for col, res in zip(cols, results):
                with col:
                    outputs.append(render_result(res, threshold))

            if len(results) == 2:
                if outputs[0][0] == outputs[1][0]:
                    st.success(f"Kedua model sepakat: {outputs[0][0]} {EMOJI.get(outputs[0][0], '')}")
                else:
                    st.warning("Kedua model memberi hasil berbeda. Coba gambar yang lebih jelas.")

        if is_new:
            for res, (label, conf) in zip(results, outputs):
                add_history(source_name, res, label, conf)
            st.session_state.last_key = key
            if all(c >= 0.95 for _, c in outputs):
                st.toast("Keyakinan sangat tinggi!", icon="🎯")

with tab_hist:
    if not st.session_state.history:
        st.markdown(
            '<div class="kosong"><div class="ikon">🕑</div><h3>Belum ada riwayat</h3>'
            "<p>Hasil prediksi akan muncul di sini.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        hist = pd.DataFrame(st.session_state.history)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total prediksi", len(hist))
        c2.metric("Rata-rata keyakinan", f"{hist['Keyakinan (%)'].mean():.1f}%")
        c3.metric("Rata-rata waktu", f"{hist['Waktu (ms)'].mean():.0f} ms")
        st.dataframe(hist, width="stretch", hide_index=True)
        b1, b2 = st.columns(2)
        b1.download_button(
            "⬇️ Unduh CSV",
            hist.to_csv(index=False).encode("utf-8"),
            "riwayat_prediksi.csv",
            "text/csv",
        )
        if b2.button("🗑️ Hapus riwayat"):
            st.session_state.history = []
            st.rerun()