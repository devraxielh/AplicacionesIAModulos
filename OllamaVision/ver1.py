from ollama import Client
import base64

client = Client()
with open('imagen.jpg', 'rb') as file:
    image_base64 = base64.b64encode(file.read()).decode('utf-8')

response = client.generate(
    model='gemma3:latest',
    prompt='¿Qué ves en esta imagen (limítate solo a lo que ves sin información extra, aunque si identificas algo conocido en la imagen mencionalo)? en 40 palabras',
    images=[image_base64]
)

print(response["response"])