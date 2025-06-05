from ollama import Client
import base64

client = Client()
with open('imagen3.png', 'rb') as file:
    image_base64 = base64.b64encode(file.read()).decode('utf-8')

response = client.generate(
    model='gemma3:latest',
    prompt='Lee y transcribe el texto en esta imagen, de manera textual. solo responde el texto de la imagen',
    images=[image_base64]
)

#print(response)
print(response["response"])