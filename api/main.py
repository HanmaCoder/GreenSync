"""
GreenSync — FastAPI Application Entry Point
"""
import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes import router
from utils.helpers import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GreenSync API",
    description="Real-Time Energy Demand Forecaster for Microgrids",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    logger.info("GreenSync API starting up...")
    # Auto-train if no model exists
    from pathlib import Path
    from config.settings import MODEL_PATH
    if not Path(MODEL_PATH).exists():
        logger.info("No trained model found — auto-training on startup...")
        try:
            from models.train_model import train
            train()
            logger.info("Auto-training complete.")
        except Exception as e:
            logger.warning(f"Auto-training failed: {e}. Use POST /api/v1/train to retry.")


if __name__ == "__main__":
    import uvicorn
    from config.settings import API_HOST, API_PORT, API_RELOAD
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)
