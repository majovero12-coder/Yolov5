import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# ---------------- CONFIGURACIÓN DE PÁGINA ----------------
st.set_page_config(
    page_title="🔍 Detección de Objetos con YOLOv5",
    page_icon="🤖",
    layout="wide"
)

# ---------------- ESTILOS PERSONALIZADOS ----------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(180deg, #f4f4ff 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 1.5rem;
    }
    h1 {
        text-align: center;
        color: #3a0ca3;
        font-weight: 800;
        font-size: 2.5rem !important;
    }
    h2, h3 {
        color: #4a4e69;
    }
    .stButton button {
        background: linear-gradient(90deg, #3a0ca3, #7209b7);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        padding: 0.6rem 1rem;
        transition: 0.3s ease;
    }
    .stButton button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #7209b7, #3a0ca3);
    }
    .stSlider [role='slider'] {
        accent-color: #7209b7;
    }
    .block-container {
        padding-top: 2rem;
    }
    .result-box {
        background-color: #f2e9e4;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 500;
        color: #22223b;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TÍTULO Y DESCRIPCIÓN ----------------
st.title("🚀 Detección de Objetos con YOLOv5")

st.markdown("""
Sube una imagen o captura desde tu cámara 📸 para que la IA detecte los objetos presentes en ella.  
Puedes ajustar los parámetros en la barra lateral para mejorar los resultados.
""")

# ---------------- FUNCIÓN DE CARGA DEL MODELO ----------------
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            model = yolov5.load(model_path)
            return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        💡 **Sugerencias:**
        - Instala versiones compatibles:
          ```
          pip install torch==1.12.0 torchvision==0.13.0 yolov5==7.0.9
          ```
        - Verifica que el archivo `yolov5s.pt` esté en la carpeta del proyecto.
        """)
        return None

# ---------------- CARGA DEL MODELO ----------------
with st.spinner("🔄 Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

# ---------------- CONFIGURACIÓN Y DETECCIÓN ----------------
if model:
    with st.sidebar:
        st.markdown("## ⚙️ Configuración de Detección")
        st.caption("Ajusta los parámetros del modelo a tu preferencia:")
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)

        st.markdown("---")
        st.markdown("### 🔧 Opciones avanzadas")
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("⚠️ Algunas opciones avanzadas no están disponibles con esta versión")

    # --- SECCIÓN PRINCIPAL ---
    st.markdown("### 📷 Captura o Sube tu Imagen")
    picture = st.camera_input("Toma una foto o sube una imagen para analizar", key="camera")

    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("🧠 Detectando objetos..."):
            try:
                results = model(cv2_img)
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()

        try:
            predictions = results.pred[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            categories = predictions[:, 5]

            col1, col2 = st.columns([1.2, 0.8])

            with col1:
                st.subheader("📊 Imagen Procesada")
                results.render()
                st.image(cv2_img, channels='BGR', use_container_width=True)

            with col2:
                st.subheader("📦 Objetos Detectados")

                label_names = model.names
                category_count = {}

                for category in categories:
                    cat_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                    category_count[cat_idx] = category_count.get(cat_idx, 0) + 1

                data = []
                for category, count in category_count.items():
                    label = label_names[category]
                    confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                    data.append({
                        "Categoría": label,
                        "Cantidad": count,
                        "Confianza Promedio": f"{confidence:.2f}"
                    })

                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                    st.bar_chart(df.set_index('Categoría')['Cantidad'])
                else:
                    st.info("🔎 No se detectaron objetos. Intenta reducir el umbral de confianza.")

        except Exception as e:
            st.error(f"Error al procesar los resultados: {str(e)}")
            st.stop()
else:
    st.error("🚫 No se pudo cargar el modelo. Revisa las dependencias e inténtalo nuevamente.")
    st.stop()

# ---------------- PIE DE PÁGINA ----------------
st.markdown("---")
st.caption("""
**💡 Desarrollado con Streamlit, PyTorch y YOLOv5.**  
Visual elegante diseñado para una experiencia más intuitiva 💜
""")

