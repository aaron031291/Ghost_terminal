
from fastapi import APIRouter

from .ingestion import router as ingestion_router

# Create main API router
api_router = APIRouter()

# Include the ingestion router
api_router.include_router(ingestion_router)

# Export the ingestion pipeline for direct use
from .ingestion import IngestionPipeline, IngestionStatus, IngestionType, TrustLevel

__all__ = [
    'api_router',
    'IngestionPipeline',
    'IngestionStatus',
    'IngestionType',
    'TrustLevel'
]
from grace_core import protocols, router