import streamlit as st
st.set_page_config(page_title="Detección de Enfermedades del Algodón", layout="centered")

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def cargar_clases():
    with open("clases.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

clases = cargar_clases()

# === Cargar modelo entrenado ===
@st.cache_resource
def cargar_modelo():
    model = tf.keras.models.load_model("modelo_algodon.h5")
    return model

model = cargar_modelo()

# === Interfaz Streamlit ===
st.title("Clasificador de Enfermedades en Hojas de Algodón")
st.markdown("Sube una imagen de una hoja de algodón para detectar su estado.")

imagen_subida = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])

if imagen_subida:
    imagen = Image.open(imagen_subida).convert("RGB")
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    # Preprocesamiento
    imagen_redimensionada = imagen.resize((224, 224))
    img_array = np.array(imagen_redimensionada) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predicción
    predicciones = model.predict(img_batch)
    indice_clase = np.argmax(predicciones[0])
    confianza = predicciones[0][indice_clase]

    # Resultado
    st.subheader("Predicción:")
    st.success(f"Clase: **{clases[indice_clase]}**")
    st.info(f"Confianza: `{confianza:.2%}`")

    # Mostrar gráfico de probabilidades
    st.subheader("Distribución de probabilidades:")
    fig, ax = plt.subplots()
    ax.bar(clases, predicciones[0])
    ax.set_ylabel("Probabilidad")
    ax.set_xticklabels(clases, rotation=45, ha="right")
    st.pyplot(fig)