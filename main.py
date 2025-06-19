from fastapi import FastAPI, Request
import requests

app = FastAPI()

TOKENIZER_URL = "https://tokenizer-server-1.onrender.com/"
MODEL_URL = "https://model-server-8.onrender.com/"

@app.get("/")
def home():
    return {"message": "LLM API Gateway is ready."}

@app.post("/generate/")
async def generate(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return {"response": "Prompt is empty."}

        # 1. Tokenize
        tokenizer_response = requests.post(f"{TOKENIZER_URL}tokenize/", json={"prompt": prompt})
        tokenizer_data = tokenizer_response.json()

        # 2. Generate
        model_response = requests.post(f"{MODEL_URL}generate/", json=tokenizer_data)
        model_output = model_response.json()

        return model_output  # This should include "output_ids" or "response"

    except Exception as e:
        return {"error": f"Exception: {str(e)}"}
