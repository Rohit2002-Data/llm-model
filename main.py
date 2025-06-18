from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").to("cpu")
model.eval()

app = FastAPI()

@app.get("/")
def home():
    return {"message": "LLM API is ready."}

@app.post("/generate/")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return {"response": "Prompt is empty."}

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": result}
