# 🚗 Detección de Placas Vehiculares

App de detección automática de placas vehiculares usando **YOLOv8** y **Streamlit**.

Desarrollado para el **Bootcamp IA Innovadora 2026 — Talento Tech**.

---

## 📁 Estructura del proyecto

```
placa-detector/
├── app.py               # Aplicación principal
├── requirements.txt     # Dependencias Python
├── runtime.txt          # Versión de Python para Streamlit Cloud
├── .gitignore
└── models/
    └── best.pt          # Modelo YOLOv8 entrenado
```

---

## 🖥️ Ejecutar localmente

**1. Clona el repositorio**
```bash
git clone https://github.com/TU_USUARIO/TU_REPO.git
cd TU_REPO
```

**2. Crea y activa el entorno virtual**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Instala las dependencias**
```bash
pip install -r requirements.txt
```

**4. Ejecuta la app**
```bash
streamlit run app.py
```

**5. Abre en tu navegador**
```
http://localhost:8501
```

---

## ☁️ Desplegar en Streamlit Cloud

**1. Sube el proyecto a GitHub**
- Crea un repositorio en [github.com](https://github.com)
- Asegúrate de incluir la carpeta `models/` con el archivo `best.pt`

**2. Crea una cuenta en Streamlit Cloud**
- Ve a [share.streamlit.io](https://share.streamlit.io)
- Inicia sesión con tu cuenta de GitHub

**3. Despliega la app**
- Haz clic en **"Create app"**
- Elige **"Deploy from GitHub"**
- Completa el formulario:
  - **Repository:** `TU_USUARIO/TU_REPO`
  - **Branch:** `main`
  - **Main file path:** `app.py`
- Haz clic en **"Advanced settings"** y selecciona **Python 3.10**
- Haz clic en **"Deploy"**

---

## ✨ Funcionalidades

- 📤 Carga de imágenes (JPG, PNG, WEBP)
- 🎯 Detección de placas con umbral de confianza ajustable
- 📦 Visualización de bounding boxes sobre la imagen original
- ✂️ Recorte automático de cada placa detectada
- ⬇️ Descarga de imagen anotada y recortes individuales
- 📊 Métricas en tiempo real (cantidad de detecciones, confianza)

---

## 🛠️ Tecnologías

| Herramienta | Versión |
|---|---|
| Python | 3.10 |
| Streamlit | 1.37.0 |
| Ultralytics YOLO | 8.3.0 |
| OpenCV | 4.10.0 |
| NumPy | ≥1.23, <2.0 |
