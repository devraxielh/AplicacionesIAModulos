import streamlit as st
import requests
st.set_page_config(page_title="Descripción de imagen con Ollama", layout="centered")
st.title("Descripción de Imagen")
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen cargada", use_container_width=True)
    if st.button("🔍 Generar descripción"):
        with st.spinner("Consultando API..."):
            try:
                files = {"image": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post("http://127.0.0.1:8000/img_txt", files=files)
                response.raise_for_status()
                resultado = response.json()
                st.success("Descripción generada:")
                st.markdown(f"**{resultado['response']}**")
            except Exception as e:
                st.error(f"Error: {e}")