
from fastapi import FastAPI, HTTPException, status
import logging
from typing import List
from .schemas import *
from .services import DocumentService
from .storage import DocumentStorage

logger = logging.getLogger(__name__)

def register_routes(app: FastAPI):
    """
    Registra las rutas directamente sobre `app` usando app.post/app.get.
    Llama a esta función desde main después de crear `app`.
    """
    storage = DocumentStorage()
    document_service = DocumentService(storage)

    @app.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, summary="Cargar documento")
    def upload_document(document: DocumentUpload):
        try:
            return document_service.upload_document(document)
        except Exception as e:
            logger.error("Error en /upload: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"error": "Error interno al cargar el documento"})

    @app.post("/generate-embeddings", response_model=EmbeddingResponse, summary="Generar embeddings")
    def generate_embeddings(request: EmbeddingRequest):
        try:
            return document_service.generate_embedding(request)
        except ValueError as e:
            logger.warning("Documento no encontrado: %s", e)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "Documento no encontrado", "document_id": request.document_id})
        except Exception as e:
            logger.error("Error en /generate-embeddings: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"error": "Error interno al generar embeddings"})

    @app.post("/search", response_model=SearchResponse, summary="Buscar documentos")
    def search_documents(query: SearchQuery):
        try:
            if _contains_sensitive_content(query.query):
                logger.warning("Consulta con contenido sensible detectado: %s", query.query[:50])
                return SearchResponse(results=[])
            return document_service.search_documents(query)
        except Exception as e:
            logger.error("Error en /search: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"error": "Error interno en la búsqueda"})

    @app.post("/ask", response_model=AnswerResponse, summary="Hacer pregunta")
    def ask_question(question: QuestionRequest):
        try:
            if _contains_inappropriate_content(question.question):
                logger.warning("Pregunta inapropiada detectada: %s", question.question[:50])
                return AnswerResponse(question=question.question, answer="No puedo responder a este tipo de consultas.", grounded=False)
            return document_service.answer_question(question)
        except Exception as e:
            logger.error("Error en /ask: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"error": "El servicio externo no pudo procesar la solicitud en este momento."})


def _contains_sensitive_content(text: str) -> bool:
    sensitive_keywords = ["contraseña", "password", "tarjeta", "credencial", "secreto"]
    return any(keyword in text.lower() for keyword in sensitive_keywords)

def _contains_inappropriate_content(text: str) -> bool:
    inappropriate_keywords = ["odio", "racista", "sexista", "insulto", "ofensivo"]
    return any(keyword in text.lower() for keyword in inappropriate_keywords)
