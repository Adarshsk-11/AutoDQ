import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as datasets_router

load_dotenv()  # no-op if .env doesn't exist -- all settings have safe defaults

app = FastAPI(
    title="AutoDQ API",
    description="A Data-Centric Framework for Evaluating Dataset Reliability and Predicting Model Performance",
    version="0.1.0",
)

# CORS origins are configurable via ALLOWED_ORIGINS (comma-separated) in
# .env; defaults to "*" for zero-config local development. Set this to your
# actual frontend origin(s) before any real deployment.
_allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
_allowed_origins = (
    ["*"] if _allowed_origins_raw.strip() == "*"
    else [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router)


@app.get("/")
def root():
    return {"status": "ok", "service": "AutoDQ API"}


@app.get("/health")
def health():
    return {"status": "healthy"}
