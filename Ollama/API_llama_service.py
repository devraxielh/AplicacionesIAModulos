from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama

app = FastAPI()

class UserInput(BaseModel):
    input: str

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/profesorfisica")
def profesorfisica(data: UserInput):
    try:
        user_input = data.input.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="No input provided")

        modelo = 'llama3.2'
        mensaje = [
            {'role': 'system', 'content': 'Eres un experto profesor de física, responde en 50 palabras'},
            {'role': 'user', 'content': user_input}
        ]
        respuesta = ollama.chat(model=modelo, messages=mensaje)
        return {"response": respuesta['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#uvicorn API_llama_service:app --reload