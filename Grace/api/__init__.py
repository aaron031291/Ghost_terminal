from fastapi import FastAPI
from Grace.api.router import router

app = FastAPI(title="Grace API", version="0.1.0")
app.include_router(router) 