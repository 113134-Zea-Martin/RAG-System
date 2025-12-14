from typing import List, Dict, Optional
import logging
import re
import os
import chromadb
import cohere
from .storage import DocumentStorage
from .schemas import *
from .config import config

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self, storage: DocumentStorage):
        self.storage = storage
        self._chroma_client = None
        self._collection = None

    def _get_chroma_collection(self, name: str = "documents"):
        if self._chroma_client is None:
            self._chroma_client = chromadb.Client()
        if self._collection is None:
            self._collection = self._chroma_client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def _get_cohere_client(self):
        api_key = config.cohere_api_key or os.environ.get("COHERE_API_KEY")
        if not api_key:
            logger.error("COHERE_API_KEY is not set")
            raise EnvironmentError("Cohere API key not configured")
        return cohere.Client(api_key)
    

    def upload_document(self, upload: DocumentUpload) -> DocumentResponse:
        document_id = self.storage.save_document(upload.title, upload.content)
        logger.info(f"Document uploaded with ID: {document_id}")
        return DocumentResponse(message="Document uploaded successfully", document_id=document_id)

    def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        document = self.storage.get_document(request.document_id)
        if not document:
            logger.error(f"Document with ID {request.document_id} not found for embedding generation")
            raise ValueError("Document not found")

        # Chunking inteligente: combina oraciones hasta un máximo de caracteres
        def smart_chunker(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
            """
            Divide el texto en chunks de tamaño máximo, respetando límites de oraciones.
            Añade overlap entre chunks para mantener contexto.
            """
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_len = len(sentence)
                
                # Si una sola oración es muy larga, dividirla por palabras
                if sentence_len > max_chars:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    
                    # Dividir oración larga por palabras
                    words = sentence.split()
                    temp_chunk = []
                    temp_length = 0
                    
                    for word in words:
                        word_len = len(word) + 1  # +1 por el espacio
                        if temp_length + word_len > max_chars and temp_chunk:
                            chunks.append(" ".join(temp_chunk))
                            # Overlap: mantener últimas palabras
                            overlap_words = int(len(temp_chunk) * 0.1)
                            temp_chunk = temp_chunk[-overlap_words:] if overlap_words > 0 else []
                            temp_length = sum(len(w) + 1 for w in temp_chunk)
                        temp_chunk.append(word)
                        temp_length += word_len
                    
                    if temp_chunk:
                        chunks.append(" ".join(temp_chunk))
                    continue
                
                # Agregar oración al chunk actual
                if current_length + sentence_len > max_chars and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Overlap: mantener última oración
                    if current_chunk:
                        overlap_text = current_chunk[-1]
                        current_chunk = [overlap_text, sentence]
                        current_length = len(overlap_text) + sentence_len + 1
                    else:
                        current_chunk = [sentence]
                        current_length = sentence_len
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_len + 1
            
            # Agregar el último chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            return [c for c in chunks if c.strip()]
        
        chunks_texts = smart_chunker(document["content"], max_chars=1000)
        
        if not chunks_texts:
            logger.error("No content to chunk for document %s", request.document_id)
            raise ValueError("Document has no content to embed")
        
        logger.info(f"Document {request.document_id} split into {len(chunks_texts)} chunks")

        # Embedding con Cohere - procesamiento por lotes
        co = self._get_cohere_client()
        
        def embed_texts_batch(texts: List[str], batch_size: int = 96) -> List[List[float]]:
            """
            Genera embeddings en lotes para evitar límites de la API.
            Cohere permite hasta 96 textos por request.
            """
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.debug(f"Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                try:
                    resp = co.embed(
                        texts=batch,
                        model="embed-multilingual-v3.0",
                        input_type="search_document",
                        embedding_types=["float"],
                    )
                    all_embeddings.extend(resp.embeddings.float)
                except Exception as e:
                    logger.error(f"Error in batch {i//batch_size + 1}: {e}")
                    raise
            return all_embeddings

        try:
            embeddings = embed_texts_batch(chunks_texts, batch_size=96)
            logger.info(f"Generated {len(embeddings)} embeddings for document {request.document_id}")
        except Exception as e:
            logger.exception("Error generating embeddings for document %s", request.document_id)
            raise

        # Preparar datos para Chroma
        metadatas = [{"source": request.document_id, "chunk_index": i} for i in range(len(chunks_texts))]
        ids = [f"{request.document_id}_chunk_{i}" for i in range(len(chunks_texts))]

        # Guardar en Chroma en lotes
        try:
            collection = self._get_chroma_collection(name="documents")
            
            # Chroma también tiene límites, procesar en lotes
            batch_size = 100
            for i in range(0, len(chunks_texts), batch_size):
                end_idx = min(i + batch_size, len(chunks_texts))
                logger.debug(f"Saving to Chroma batch {i//batch_size + 1}/{(len(chunks_texts)-1)//batch_size + 1}")
                collection.add(
                    documents=chunks_texts[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
            logger.info(f"Saved {len(chunks_texts)} chunks to Chroma for document {request.document_id}")
        except Exception as e:
            logger.exception("Error saving embeddings to Chroma for document %s", request.document_id)
            raise

        try:
            self.storage.create_embedding(request.document_id, document["content"])
        except Exception:
            logger.debug("storage.create_embedding falló (no crítico) para %s", request.document_id)

        logger.info(f"Embedding generated and stored for document ID: {request.document_id} ({len(chunks_texts)} chunks)")
        return EmbeddingResponse(message=f"Embedding generated successfully ({len(chunks_texts)} chunks)")

    def search_documents(self, query: SearchQuery) -> SearchResponse:
        """
        Busca por similitud: genera embedding de la query, recupera chunks de Chroma
        y normaliza el resultado a SearchResponse.
        """
        co = self._get_cohere_client()
        try:
            q_embed = co.embed(
                texts=[query.query],
                model="embed-multilingual-v3.0",
                # model="command-a-translate-08-2025",
                input_type="search_query",
                embedding_types=["float"]
            ).embeddings.float[0]
        except Exception:
            logger.exception("Error al generar embedding de la query")
            raise

        collection = self._get_chroma_collection(name="documents")
        try:
            results = collection.query(
                query_embeddings=[q_embed],
                n_results=getattr(query, "k", 5),
                include=["documents", "metadatas", "distances"]
            )
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0] if "ids" in results else [None] * len(docs)
            distances = results.get("distances", [[]])[0] if "distances" in results else None
        except Exception:
            logger.exception("Error consultando Chroma")
            raise

        seen = set()
        items = []
        for i, doc_text in enumerate(docs):
            raw_id = ids[i] if i < len(ids) else None
            meta = metadatas[i] if i < len(metadatas) else {}

            dedup_key = raw_id or (doc_text[:200] if isinstance(doc_text, str) else f"chunk_{i}")
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            document_id = None
            if isinstance(meta, dict) and meta.get("source"):
                document_id = str(meta.get("source"))
            elif raw_id and isinstance(raw_id, str) and "_chunk_" in raw_id:
                document_id = raw_id.split("_chunk_")[0]
            elif raw_id:
                document_id = str(raw_id)

            content_snippet = (doc_text[:300]) if isinstance(doc_text, str) else ""

            title = ""
            if document_id:
                stored = self.storage.get_document(document_id)
                if stored:
                    title = stored.get("title", "") or ""

            similarity_score = 0.0
            if distances and i < len(distances):
                try:
                    dist = float(distances[i])
                    similarity_score = max(0.0, min(1.0, 1.0 - dist))
                except Exception:
                    similarity_score = 0.0
            else:
                if isinstance(meta, dict) and "score" in meta:
                    try:
                        similarity_score = float(meta.get("score", 0.0))
                        similarity_score = max(0.0, min(1.0, similarity_score))
                    except Exception:
                        similarity_score = 0.0

            items.append({
                "document_id": document_id or (raw_id or f"chunk_{i}"),
                "title": title,
                "content_snippet": content_snippet,
                "similarity_score": similarity_score,
                "chunk_id": raw_id,
                "chunk_index": meta.get("chunk_index") if isinstance(meta, dict) else None
            })

        logger.info("Search performed for query: %s with %d results", query.query, len(items))
        return SearchResponse(results=items)


    def answer_question(self, request: QuestionRequest) -> AnswerResponse:
        """
        RAG: embed pregunta, recuperar chunks de Chroma y generar respuesta con Cohere.
        Devuelve AnswerResponse con context_used como lista de dicts compatibles con AnswerContext.
        """
        co = self._get_cohere_client()

        # 1) Embedding de la pregunta
        try:
            q_embed = co.embed(
                texts=[request.question],
                model="embed-multilingual-v3.0",
                input_type="search_query",
                embedding_types=["float"]
            ).embeddings.float[0]
        except Exception:
            logger.exception("Error al generar embedding de la pregunta")
            raise

        # 2) Retrieve desde Chroma
        collection = self._get_chroma_collection(name="documents")
        try:
            results = collection.query(
                query_embeddings=[q_embed],
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0] if "distances" in results else None
            ids = results.get("ids", [[]])[0] if "ids" in results else [None] * len(docs)
        except Exception:
            logger.exception("Error consultando Chroma para RAG")
            raise

        if not docs:
            return AnswerResponse(
                question=request.question,
                answer="Lo siento, no tengo esa información en mis historias. 📚😔",
                context_used=None,
                grounded=False
            )

        # 3) Construir contexto estructurado (AnswerContext-like)
        context_used = []
        for i, doc_text in enumerate(docs):
            meta = metadatas[i] if i < len(metadatas) else {}
            raw_id = ids[i] if i < len(ids) else None
            dist = distances[i] if distances and i < len(distances) else None

            # Extraer document_id
            document_id = None
            if isinstance(meta, dict) and meta.get("source"):
                document_id = str(meta.get("source"))
            elif raw_id and isinstance(raw_id, str) and "_chunk_" in raw_id:
                document_id = raw_id.split("_chunk_")[0]
            elif raw_id:
                document_id = str(raw_id)

            # snippet = el texto del chunk recuperado
            content_snippet = doc_text if isinstance(doc_text, str) else ""

            # similarity_score derivado de distance si existe, sino fallback a meta.score
            similarity_score = 0.0
            if dist is not None:
                try:
                    similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))
                except Exception:
                    similarity_score = 0.0
            else:
                if isinstance(meta, dict) and "score" in meta:
                    try:
                        similarity_score = float(meta.get("score", 0.0))
                        similarity_score = max(0.0, min(1.0, similarity_score))
                    except Exception:
                        similarity_score = 0.0

            context_used.append({
                "document_id": document_id,
                "chunk_id": raw_id,
                "chunk_index": meta.get("chunk_index") if isinstance(meta, dict) else None,
                "content_snippet": content_snippet,
                "similarity_score": similarity_score
            })

        # 4) Construir prompt con el contexto (usar solo los snippets)
        context_text = "\n\n".join([f"{i+1}. {c['content_snippet']}" for i, c in enumerate(context_used)])

        system = (
            "Eres un asistente amable y servicial especializado en contar historias infantiles. "
            "Tus respuestas deben ser amigables y con tono entusiasta, máximo 3 oraciones, en español. "
            "Utiliza solo el contenido de las historias para responder. "
            "Si no sabes, responde: \"Lo siento, no tengo esa información en mis historias. 📚😔\""
        )

        prompt = f"""{system}

Aquí tienes partes relevantes de las historias:
{context_text}

Pregunta del usuario: "{request.question}"

Responde siguiendo TODAS las reglas.
"""

        # 5) Llamar a la API de chat de Cohere para generar la respuesta
        try:
            chat_resp = co.chat(model="command-a-translate-08-2025", message=prompt, temperature=0.3)
            answer_text = chat_resp.text.strip()
        except Exception:
            logger.exception("Error llamando a la API de chat de Cohere")
            raise

        grounded = len(context_used) > 0
        logger.info("Question answered: %s | Grounded: %s", request.question, grounded)

        return AnswerResponse(
            question=request.question,
            answer=answer_text,
            context_used=context_used,
            grounded=grounded
        )