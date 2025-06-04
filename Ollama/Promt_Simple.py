import ollama
modelo = 'llama3'
mensajes = [
    {'role': 'user', 'content': '¿Por qué el cielo es azul? Responde en 20 palabras.'}
]
respuesta = ollama.chat(model=modelo, messages=mensajes)
print(respuesta['message']['content'])