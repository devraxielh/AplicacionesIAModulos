import streamlit as st
import ollama
import json

# Cargar base de conocimiento desde JSON
with open("cartagena_kb.json", encoding="utf-8") as f:
    CARTAGENA_KB = json.load(f)

def format_context(user_input):
    return f"""Eres un guía turístico experto en Cartagena, Colombia.
    Usa esta información para responder: {json.dumps(CARTAGENA_KB, ensure_ascii=False)}
    Pregunta del turista: {user_input}
    Responde de manera amable y profesional, usando la información proporcionada y tu conocimiento general sobre Cartagena."""

def get_response(message):
    response = ollama.chat(model='gemma3:1b', messages=[
        {'role': 'system', 'content': format_context(message)},
        {'role': 'user', 'content': message}
    ])
    return response['message']['content']

st.title("🏖️ Guía Turístico de Cartagena")
st.markdown("Pregúntame sobre lugares turísticos, historia, gastronomía y más de Cartagena.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué quieres saber sobre Cartagena?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_response(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})