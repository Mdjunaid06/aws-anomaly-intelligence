"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.database import init_db
from .routes import anomalies, batch, explanations, health, observations, replay, stations, system

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting AWS Anomaly Intelligence Backend")
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AWS Anomaly Intelligence Backend")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    debug=settings.debug,
    lifespan=lifespan,
)

# Add CORS middleware
allowed_origins = {
    origin.strip()
    for origin in (
        [settings.frontend_origin]
        + ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    if origin and origin.strip()
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(observations.router)
app.include_router(anomalies.router)
app.include_router(batch.router)
app.include_router(health.router)
app.include_router(replay.router)
app.include_router(explanations.router)
app.include_router(stations.router)
app.include_router(system.router)


@app.get("/", tags=["root"])
def root():
    """Root endpoint."""
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "environment": settings.environment,
        "status": "running",
    }


@app.get("/health", tags=["root"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.api_title,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
