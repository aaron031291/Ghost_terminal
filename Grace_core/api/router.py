from fastapi import FastAPI
from Grace.main import main

app = FastAPI(title="Grace API", version="0.1.0")

@app.get("/health")
def health_check():
    return {"status": "Grace is alive."}

@app.post("/run")
def run_grace():
    result = main()
    return {"result": result}