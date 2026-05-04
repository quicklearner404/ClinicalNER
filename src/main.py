from fastapi import FastAPI
from src.inference import run_prediction

app = FastAPI(title="Clinical NER Service")

@app.get("/")
def home():
    return {"status": "Clinical NER Agent is Online"}

@app.post("/extract")
def extract_entities(data: dict):
    text = data.get("text", "")
    if not text:
        return {"entities": []}
    
    # Get all entities from the model
    all_entities = run_prediction(text)
    
    # SYSTEM REQUIREMENT: Confidence must be at least 60%
    # Also filtering out common non-medical noise
    blacklist = ["patient", "health", "hospital"]
    
    filtered = [
        ent for ent in all_entities 
        if ent['score'] >= 0.60 and ent['word'].lower().strip() not in blacklist
    ]
    
    return {"entities": filtered}