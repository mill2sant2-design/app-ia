import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
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
    .plate-text {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 6px;
        color: #1a1a2e;
        text-align: center;
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 10px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

LETRA_A_NUMERO = {"O": "0", "I": "1", "G": "6", "B": "8", "S": "5", "Z": "2"}
NUMERO_A_LETRA = {"0": "O", "1": "I", "6": "G", "8": "B", "5": "S", "2": "Z"}

def corregir_placa(texto):
    texto = "".join(c for c in texto if c.isalnum()).upper()
    if len(texto) != 6:
        return texto
    resultado = []
    for i, c in enumerate(texto):
        if i < 3:
            resultado.append(NUMERO_A_LETRA.get(c, c))
        else:
            resultado.append(LETRA_A_NUMERO.get(c, c))
    return "".join(resultado)

def preprocesar(imagen_pil):
    w, h = imagen_pil.size
    imagen_pil = imagen_pil.resize((w * 4, h * 4), Image.LANCZOS)
    gris = imagen_pil.convert("L")
    gris = ImageEnhance.Contrast(gris).enhance(2.5)
    gris = gris.filter(ImageFilter.SHARPEN)
    return gris

def leer_placa(imagen_pil):
    img = preprocesar(imagen_pil)

    configs = [
        "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ]

    candidatos = []
    for cfg in configs:
        try:
            raw = pytesseract.image_to_string(img, config=cfg).strip()
            limpio = "".join(c for c in raw if c.isalnum()).upper()
            if limpio:
                candidatos.append(limpio)
        except:
            continue

    mejor = min(candidatos, key=lambda x: abs(len(x) - 6), default="")

    if len(mejor) == 6:
        return corregir_placa(mejor)
    if len(mejor) > 6:
        return corregir_placa(mejor[:6])
    return mejor

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=80)
    st.markdown("## ⚙️ Configuración")
    conf_threshold = st.slider("Umbral de confianza", 0.0, 1.0, 0.4, 0.05)
    iou_threshold = st.slider("Umbral IoU (NMS)", 0.0, 1.0, 0.30, 0.05)
    show_conf = st.toggle("Mostrar confianza en etiquetas", value=True)
    st.markdown("---")
    st.markdown("### 📋 Acerca del modelo")
    st.info("**Tarea:** Detección de placas\n\n**Clases:** `placa`\n\n**Arquitectura:** YOLOv8\n\n**Parámetros:** 3M")

st.markdown('<p class="title">🚗 Detección de Placas Vehiculares</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sube una imagen para detectar, recortar y leer placas automáticamente.</p>', unsafe_allow_html=True)
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

                with st.spinner("📖 Leyendo texto..."):
                    texto = leer_placa(cropped_pil)

                if texto:
                    display = f"{texto[:3]}-{texto[3:]}" if len(texto) == 6 else texto
                    st.markdown(f'<div class="plate-text">🔤 {display}</div>', unsafe_allow_html=True)
                else:
                    st.info("No se pudo leer el texto.")

                buf_crop = io.BytesIO()
                cropped_pil.save(buf_crop, format="PNG")
                st.download_button(f"⬇️ Descargar placa {i+1}",
