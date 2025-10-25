import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# ==========================
# CONFIGURACIÓN DE PÁGINA
# ==========================
st.set_page_config(
    page_title="🔍 Detección de Objetos con IA",
    page_icon="🤖",
    layout="wide"
)

# ==========================
# ESTILO PERSONALIZADO
# ==========================
st.markdown("""
    <style>
        /* Fondo degradado */
        .stApp {
            background: linear-gradient(135deg, #141e30, #243b55);
            color: white;
            font-family: 'Poppins', sans-serif;
        }

        /* Título principal */
        h1 {
            text-align: center;
            font-size: 2.7em !important;
            color: #ffffff;
            text-shadow: 2px 2px 15px rgba(0, 200, 255, 0.4);
            margin-bottom: 0.5em;
        }

        /* Subtítulos */
        h2, h3, h4 {
            color: #8be9fd !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.07);
            backdrop-filter: blur(10px);
            border-right: 2px solid rgba(255,255,255,0.1);
        }

        /* Botones */
        button[kind="primary"], button[kind="secondary"] {
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            color: white !important;
            border-radius: 10px !important;
            font-weight: bold;
            border: none !important;
            transition: all 0.3s ease-in-out;
        }

        button[kind="primary"]:hover, button[kind="secondary"]:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #0072ff, #00c6ff);
        }

        /* Cuadros de texto y sliders */
        .stSlider, textarea, input {
            color: white !important;
        }

        div[data-testid="stMarkdownContainer"] p {
            color: #e6f7ff !important;
        }

        /* Tablas */
        .stDataFrame {
            background-color: rgba(255,255,255,0.05) !important;
            border-radius: 10px;
        }

        /* Gráficos */
        .stPlotlyChart, .stBarChart {
            background-color: transparent !important;
        }

        /* Pie de página */
        footer, .stApp [data-testid="stFooter"] {
            visibility: hidden;
        }

        /* Spinner */
        .stSpinner > div > div {
            border-top-color: #00c6ff !important;
        }

    </style>
""", unsafe_allow_html=True)

# ==========================
# TÍTULO PRINCIPAL
# ==========================
st.title("🤖 Detección de Objetos en Imágenes 🔍")
st.markdown("""
Esta aplicación utiliza **YOLOv5** para detectar objetos en imágenes tomadas con tu cámara.  
Ajusta los parámetros en la barra lateral para personalizar la precisión de la detección.
""")

# ==========================
# FUNCIÓN PARA CARGAR MODELO YOLOV5
# ==========================
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception:
                st.warning("Método alternativo de carga activado...")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        **Recomendaciones:**
        1. Instala versiones compatibles:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. Asegúrate de tener el archivo `yolov5s.pt` en la carpeta correcta.
        """)
        return None

# ==========================
# CARGA DEL MODELO
# ==========================
with st.spinner("⚙️ Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

# ==========================
# INTERFAZ PRINCIPAL
# ==========================
if model:
    st.sidebar.title("⚙️ Panel de Configuración")

    with st.sidebar:
        st.subheader("🎯 Parámetros de detección")
        model.conf = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

        st.subheader("🔧 Opciones avanzadas")
        try:
            model.agnostic = st.checkbox("NMS class-agnostic", False)
            model.multi_label = st.checkbox("Múltiples etiquetas por caja", False)
            model.max_det = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones no están disponibles en esta versión")

    # ==========================
    # CAPTURA Y PROCESO DE IMAGEN
    # ==========================
    picture = st.camera_input("📸 Captura una imagen con tu cámara", key="camera")

    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("🔍 Analizando objetos en la imagen..."):
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

            col1, col2 = st.columns([3, 2])

            with col1:
                st.subheader("🖼️ Imagen con detecciones")
                results.render()
                st.image(cv2_img, channels="BGR", use_container_width=True)

            with col2:
                st.subheader("📋 Resultados de detección")

                label_names = model.names
                category_count = {}

                for category in categories:
                    idx = int(category.item()) if hasattr(category, 'item') else int(category)
                    category_count[idx] = category_count.get(idx, 0) + 1

                data = []
                for category, count in category_count.items():
                    label = label_names[category]
                    confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                    data.append({
                        "Categoría": label,
                        "Cantidad": count,
                        "Confianza promedio": f"{confidence:.2f}"
                    })

                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                    st.bar_chart(df.set_index("Categoría")["Cantidad"])
                else:
                    st.info("No se detectaron objetos. Intenta reducir el umbral de confianza.")

        except Exception as e:
            st.error(f"Error al procesar los resultados: {str(e)}")
else:
    st.error("❌ No se pudo cargar el modelo YOLOv5 correctamente.")
    st.stop()

# ==========================
# PIE DE PÁGINA
# ==========================
st.markdown("---")
st.caption("""
💡 *Desarrollado con Streamlit, PyTorch y YOLOv5.*  
Creado para demostraciones de detección de objetos en tiempo real.
""")
