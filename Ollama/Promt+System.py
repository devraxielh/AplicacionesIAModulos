import ollama
modelo = 'llama3'
mensajes = [
    {'role': 'system', 'content': 'Eres un experto en física capaz de responder preguntas complejas de manera clara y precisa.responde en 20 palabras'},
    {'role': 'user', 'content': '¿Por qué el cielo es azul?'}
]
respuesta = ollama.chat(model=modelo, messages=mensajes)
print(respuesta['message']['content'])