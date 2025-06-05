from ollama import Client
import base64

client = Client()
with open('imagen2.jpg', 'rb') as file:
    image_base64 = base64.b64encode(file.read()).decode('utf-8')

response = client.generate(
    model='gemma3:latest',
    prompt='Enumera todos los objetos que ves en la imagen, en 20 palabras solo responde con los objetos',
    images=[image_base64]
)

print(response["response"])