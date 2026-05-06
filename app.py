import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Detección de Placas Vehiculares",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Estilos
# ─────────────────────────────────────────────
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
    .plate-box {
        border: 2px solid #4361ee;
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 10px;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Cargar modelo (cacheado)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=80)
    st.markdown("## ⚙️ Configuración")

    conf_threshold = st.slider(
        "Umbral de confianza",
        min_value=0.0, max_value=1.0,
        value=0.5, step=0.05,
        help="Detecciones con confianza por debajo de este valor serán ignoradas."
    )

    iou_threshold = st.slider(
        "Umbral IoU (NMS)",
        min_value=0.0, max_value=1.0,
        value=0.45, step=0.05,
        help="Controla la supresión de detecciones solapadas."
    )

    show_conf = st.toggle("Mostrar confianza en etiquetas", value=True)

    st.markdown("---")
    st.markdown("### 📋 Acerca del modelo")
    st.info(
        "**Tarea:** Detección de placas\n\n"
        "**Clases:** `placa`\n\n"
        "**Arquitectura:** YOLOv8\n\n"
        "**Parámetros:** 3M"
    )

# ─────────────────────────────────────────────
# Cabecera principal
# ─────────────────────────────────────────────
st.markdown('<p class="title">🚗 Detección de Placas Vehiculares</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sube una imagen para detectar y recortar placas automáticamente.</p>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# Subida de imagen
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Selecciona una imagen",
    type=["jpg", "jpeg", "png", "webp"],
    help="Formatos soportados: JPG, PNG, WEBP"
)

if uploaded_file is not None:
    # Leer imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ No se pudo leer la imagen. Intenta con otro archivo.")
        st.stop()

    h, w, _ = image.shape

    # ── Predicción ──────────────────────────────
    with st.spinner("🔍 Analizando imagen..."):
        results = model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

    boxes = results[0].boxes
    num_detections = len(boxes) if boxes is not None else 0

    # ── Métricas ─────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>{num_detections}</h3><p>Placas detectadas</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>{w} × {h}</h3><p>Resolución original</p></div>', unsafe_allow_html=True)
    with c3:
        best_conf = float(boxes.conf.max()) if num_detections > 0 else 0.0
        st.markdown(f'<div class="metric-card"><h3>{best_conf:.0%}</h3><p>Mayor confianza</p></div>', unsafe_allow_html=True)

    st.divider()

    # ── Imagen anotada ────────────────────────────
    annotated = results[0].plot(conf=show_conf)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📌 Detecciones sobre la imagen")
        st.image(annotated, channels="BGR", use_container_width=True)

        # Botón de descarga de imagen anotada
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_annotated = Image.fromarray(annotated_rgb)
        buf = io.BytesIO()
        pil_annotated.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Descargar imagen anotada",
            data=buf.getvalue(),
            file_name="deteccion_placas.png",
            mime="image/png"
        )

    with col2:
        st.subheader("🔍 Placas recortadas")

        if num_detections > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                cropped = image[y1:y2, x1:x2]
                conf_val = float(box.conf[0])

                # Mostrar recorte
                st.markdown(f'<div class="plate-box">', unsafe_allow_html=True)
                st.image(cropped, channels="BGR", use_container_width=True,
                         caption=f"Placa {i+1} — Confianza: {conf_val:.1%}")

                # Descarga individual
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(cropped_rgb)
                buf_crop = io.BytesIO()
                pil_crop.save(buf_crop, format="PNG")
                st.download_button(
                    label=f"⬇️ Descargar placa {i+1}",
                    data=buf_crop.getvalue(),
                    file_name=f"placa_{i+1}.png",
                    mime="image/png",
                    key=f"dl_{i}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ No se detectaron placas. Prueba bajando el umbral de confianza.")

else:
    # Estado vacío
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #aaa;">
        <div style="font-size: 4rem;">📷</div>
        <p style="font-size: 1.1rem;">Sube una imagen para comenzar</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Talento Tech 2026 · Bootcamp IA Innovadora · Detección de Placas con YOLOv8"
    "</p>",
    unsafe_allow_html=True
)
