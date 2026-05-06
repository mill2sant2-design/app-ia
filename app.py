import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

st.set_page_config(
    page_title="Detección de Placas Vehiculares",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .subtitle { color: #6c757d; margin-top: -10px; margin-bottom: 20px; }
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 12px 18px;
        text-align: center;
        border-left: 4px solid #4361ee;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=80)
    st.markdown("## ⚙️ Configuración")
    conf_threshold = st.slider("Umbral de confianza", 0.0, 1.0, 0.5, 0.05)
    iou_threshold = st.slider("Umbral IoU (NMS)", 0.0, 1.0, 0.45, 0.05)
    show_conf = st.toggle("Mostrar confianza en etiquetas", value=True)
    st.markdown("---")
    st.markdown("### 📋 Acerca del modelo")
    st.info("**Tarea:** Detección de placas\n\n**Clases:** `placa`\n\n**Arquitectura:** YOLOv8\n\n**Parámetros:** 3M")

st.markdown('<p class="title">🚗 Detección de Placas Vehiculares</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sube una imagen para detectar y recortar placas automáticamente.</p>', unsafe_allow_html=True)
st.divider()

uploaded_file = st.file_uploader("📁 Selecciona una imagen", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_image)
    w, h = pil_image.size

    with st.spinner("🔍 Analizando imagen..."):
        results = model.predict(image_np, conf=conf_threshold, iou=iou_threshold, verbose=False)

    boxes = results[0].boxes
    num_detections = len(boxes) if boxes is not None else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>{num_detections}</h3><p>Placas detectadas</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>{w} × {h}</h3><p>Resolución original</p></div>', unsafe_allow_html=True)
    with c3:
        best_conf = float(boxes.conf.max()) if num_detections > 0 else 0.0
        st.markdown(f'<div class="metric-card"><h3>{best_conf:.0%}</h3><p>Mayor confianza</p></div>', unsafe_allow_html=True)

    st.divider()

    annotated_np = results[0].plot(conf=show_conf)
    annotated_pil = Image.fromarray(annotated_np[..., ::-1])

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📌 Detecciones sobre la imagen")
        st.image(annotated_pil, use_container_width=True)
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        st.download_button("⬇️ Descargar imagen anotada", buf.getvalue(), "deteccion_placas.png", "image/png")

    with col2:
        st.subheader("🔍 Placas recortadas")
        if num_detections > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cropped_pil = pil_image.crop((x1, y1, x2, y2))
                conf_val = float(box.conf[0])
                st.image(cropped_pil, use_container_width=True, caption=f"Placa {i+1} — Confianza: {conf_val:.1%}")
                buf_crop = io.BytesIO()
                cropped_pil.save(buf_crop, format="PNG")
                st.download_button(f"⬇️ Descargar placa {i+1}", buf_crop.getvalue(), f"placa_{i+1}.png", "image/png", key=f"dl_{i}")
        else:
            st.warning("⚠️ No se detectaron placas. Prueba bajando el umbral de confianza.")
else:
    st.markdown('<div style="text-align:center;padding:60px 20px;color:#aaa;"><div style="font-size:4rem;">📷</div><p>Sube una imagen para comenzar</p></div>', unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align:center;color:#aaa;font-size:0.85rem;'>Talento Tech 2026 · Bootcamp IA Innovadora · Detección de Placas con YOLOv8</p>", unsafe_allow_html=True)
