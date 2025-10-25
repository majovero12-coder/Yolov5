import os
import streamlit as st
import base64
from openai import OpenAI

# ===== CONFIGURACIÓN DE PÁGINA =====
st.set_page_config(
    page_title="🔍 Análisis de Imagen con IA",
    layout="centered",
    page_icon="🤖"
)

# ===== ESTILO PERSONALIZADO =====
st.markdown("""
    <style>
        /* Fondo degradado */
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
            font-family: 'Poppins', sans-serif;
        }

        /* Título principal */
        h1 {
            text-align: center;
            font-size: 2.5em !important;
            color: #f8f9fa;
            text-shadow: 2px 2px 10px rgba(255, 255, 255, 0.2);
            margin-bottom: 20px;
        }

        /* Caja del uploader */
        div[data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 15px;
        }

        /* Botones */
        button[kind="secondary"] {
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            color: white !important;
            border-radius: 8px !important;
            font-weight: bold;
        }

        /* Caja de texto */
        textarea, input {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 8px !important;
        }

        /* Expander */
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }

        /* Spinner */
        .stSpinner > div > div {
            border-top-color: #00c6ff !important;
        }

        /* Texto de advertencia */
        .stAlert {
            background-color: rgba(255, 255, 255, 0.1);
            color: #fff !important;
        }
    </style>
""", unsafe_allow_html=True)


# ===== INTERFAZ PRINCIPAL =====
st.title("🤖 Análisis de Imagen con Inteligencia Artificial 🏞️")
st.write("Sube una imagen y deja que la IA te diga qué ve. Puedes agregar contexto para obtener una descripción más precisa.")

# ===== API KEY =====
ke = st.text_input("🔑 Ingresa tu Clave API de OpenAI", type="password")
os.environ['OPENAI_API_KEY'] = ke

api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=api_key)

# ===== SUBIR IMAGEN =====
uploaded_file = st.file_uploader("📸 Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with st.expander("👀 Vista previa de la imagen", expanded=True):
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

# ===== DETALLES ADICIONALES =====
show_details = st.toggle("📝 Añadir detalles o contexto", value=False)

if show_details:
    additional_details = st.text_area("Escribe aquí el contexto adicional:")

# ===== BOTÓN DE ANÁLISIS =====
analyze_button = st.button("🚀 Analizar Imagen")

# ===== FUNCIÓN DE ENCODE =====
def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode("utf-8")

# ===== PROCESO DE ANÁLISIS =====
if uploaded_file is not None and api_key and analyze_button:
    with st.spinner("🔍 Analizando la imagen..."):
        base64_image = encode_image(uploaded_file)
        prompt_text = "Describe detalladamente lo que ves en la imagen en español."

        if show_details and additional_details:
            prompt_text += f"\n\nContexto adicional proporcionado por el usuario:\n{additional_details}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ],
            }
        ]

        try:
            full_response = ""
            message_placeholder = st.empty()
            for completion in client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1200,
                stream=True
            ):
                if completion.choices[0].delta.content is not None:
                    full_response += completion.choices[0].delta.content
                    message_placeholder.markdown(
                        f"<div style='color:#00c6ff; font-size:1.1em'>{full_response}▌</div>",
                        unsafe_allow_html=True
                    )
            message_placeholder.markdown(
                f"<div style='color:#00c6ff; font-size:1.1em'>{full_response}</div>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

else:
    if not uploaded_file and analyze_button:
        st.warning("⚠️ Por favor, sube una imagen antes de analizar.")
    if not api_key:
        st.warning("🔑 Ingresa tu API Key para continuar.")


