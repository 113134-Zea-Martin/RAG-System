from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

from .endpoints import register_routes

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Get Talent RAG API",
    description="Sistema RAG para búsqueda semántica y generación de respuestas",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

register_routes(app)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para loguear todas las peticiones"""
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Error interno del servidor"}
        )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Recurso no encontrado"}
    )


# Endpoint de salud
@app.get("/", tags=["Health"])
async def root():
    """Endpoint de verificación de salud"""
    return {
        "message": "Get Talent RAG API está funcionando",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /api/v1/upload",
            "generate_embeddings": "POST /api/v1/generate-embeddings",
            "search": "POST /api/v1/search",
            "ask": "POST /api/v1/ask"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Endpoint de verificación de salud"""
    return {"status": "healthy", "service": "rag-api"}

if __name__ == "__main__":
    uvicorn.run(
        "Challenge.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )