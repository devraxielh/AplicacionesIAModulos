from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import ollama
from ollama import Client
import base64

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"status": "running"}

@app.post("/img_txt")
async def img_txt(image: UploadFile = File(...)):
    try:
        if image.content_type.split('/')[0] != 'image':
            raise HTTPException(status_code=400, detail="File is not an image")
        image_data = await image.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        prompt = "transcribe el texto en esta imagen, de manera textual."

        cliente = Client()
        modelo = "gemma3:latest"
        respuesta = cliente.generate(
            model=modelo,
            prompt=prompt,
            images=[encoded_image]
        )
        return {"response": respuesta["response"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#uvicorn llama_service:app --reload