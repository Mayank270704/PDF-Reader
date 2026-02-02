from fastapi import FastAPI
from app.api import router

app = FastAPI(title="AI Document Intelligence System")

app.include_router(router)
