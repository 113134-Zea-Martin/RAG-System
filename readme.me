# RAG System - Sistema de Recuperación y Generación Aumentada

Sistema RAG (Retrieval-Augmented Generation) completo que permite cargar documentos, generar embeddings semánticos, realizar búsquedas por similitud y responder preguntas usando el contexto de los documentos.

## 🚀 Características

- **Carga de documentos**: Soporta documentos grandes (hasta 200k+ caracteres)
- **Chunking inteligente**: División automática de documentos con overlap contextual
- **Embeddings semánticos**: Utiliza Cohere embed-multilingual-v3.0
- **Base de datos vectorial**: ChromaDB para almacenamiento y búsqueda eficiente
- **Búsqueda semántica**: Encuentra contenido relevante por similitud
- **Generación de respuestas**: RAG completo con Cohere para responder preguntas
- **API REST**: FastAPI con documentación automática
- **Procesamiento por lotes**: Maneja documentos grandes sin límites de API

## 📋 Requisitos

- Python 3.8+
- API Key de Cohere (obtener en [cohere.com](https://cohere.com))

## 🔧 Instalación

1. **Clonar el repositorio**
```bash
git clone <repository-url>
cd RAG-System
```

2. **Crear entorno virtual**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirementsCh.txt
```

4. **Configurar variables de entorno**

Crear archivo `.env` en la carpeta `Challenge/`:
```env
COHERE_API_KEY=tu_api_key_aqui
```

## 🎯 Uso

### Iniciar el servidor

```bash
# Desde la raíz del proyecto
python -m uvicorn Challenge.main:app --reload --host 0.0.0.0 --port 8000
```

El servidor estará disponible en `http://localhost:8000`

### Documentación interactiva

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### 1. Cargar Documento

**POST** `/upload`

Carga un documento al sistema.

```json
{
  "title": "Mi Documento",
  "content": "Contenido del documento..."
}
```

**Respuesta:**
```json
{
  "message": "Document uploaded successfully",
  "document_id": "uuid-generado"
}
```

### 2. Generar Embeddings

**POST** `/generate-embeddings`

Procesa el documento y genera embeddings vectoriales.

```json
{
  "document_id": "uuid-del-documento"
}
```

**Respuesta:**
```json
{
  "message": "Embedding generated successfully (200 chunks)"
}
```

**Características:**
- Divide documentos en chunks de ~1000 caracteres
- Respeta límites de oraciones
- Añade overlap del 10% entre chunks
- Procesa en lotes de 96 textos (límite de Cohere)
- Guarda en ChromaDB en lotes de 100

### 3. Buscar Documentos

**POST** `/search`

Búsqueda semántica por similitud.

```json
{
  "query": "¿Qué información hay sobre...?"
}
```

**Respuesta:**
```json
{
  "results": [
    {
      "document_id": "uuid",
      "title": "Título del documento",
      "content_snippet": "Fragmento relevante...",
      "similarity_score": 0.95
    }
  ]
}
```

### 4. Hacer Pregunta (RAG)

**POST** `/ask`

Responde preguntas usando el contexto de los documentos.

```json
{
  "question": "¿Cuál es el tema principal?"
}
```

**Respuesta:**
```json
{
  "question": "¿Cuál es el tema principal?",
  "answer": "Según los documentos, el tema principal es...",
  "context_used": [
    {
      "document_id": "uuid",
      "content_snippet": "Contexto usado...",
      "similarity_score": 0.92
    }
  ],
  "grounded": true
}
```

## 📁 Estructura del Proyecto

```
RAG-System/
├── Challenge/
│   ├── __init__.py
│   ├── config.py           # Configuración y variables de entorno
│   ├── endpoints.py        # Definición de rutas API
│   ├── main.py            # Aplicación FastAPI principal
│   ├── schemas.py         # Modelos Pydantic (request/response)
│   ├── services.py        # Lógica de negocio (RAG, embeddings)
│   ├── storage.py         # Almacenamiento de documentos
│   └── .env              # Variables de entorno (no commitear)
├── requirementsCh.txt     # Dependencias Python
└── readme.me             # Este archivo
```

## 🔄 Flujo de Trabajo

1. **Cargar documento** → `/upload`
2. **Generar embeddings** → `/generate-embeddings` con el `document_id`
3. **Buscar o preguntar**:
   - Búsqueda: `/search` con una query
   - RAG: `/ask` con una pregunta

## 🛠️ Tecnologías

- **FastAPI**: Framework web moderno y rápido
- **ChromaDB**: Base de datos vectorial embeddings
- **Cohere**: API de embeddings y LLM
  - `embed-multilingual-v3.0`: Generación de embeddings
  - `command-a-translate-08-2025`: Generación de respuestas
- **Pydantic**: Validación de datos
- **Uvicorn**: Servidor ASGI

## ⚙️ Configuración Avanzada

### Chunking

El sistema divide documentos usando estos parámetros (modificables en [services.py](Challenge/services.py)):

```python
max_chars = 1000      # Tamaño máximo de chunk
overlap = 10%         # Overlap entre chunks
```

### Límites de API

- **Cohere embeddings**: 96 textos por batch
- **ChromaDB inserts**: 100 documentos por batch

### Logging

El sistema registra:
- Peticiones HTTP
- Proceso de chunking
- Generación de embeddings
- Búsquedas y consultas
- Errores y excepciones

Nivel de log configurable en [main.py](Challenge/main.py).

## 🐛 Resolución de Problemas

### Error: "COHERE_API_KEY is not set"
- Verificar que el archivo `.env` existe en `Challenge/`
- Confirmar que la API key es válida

### Error al procesar documentos grandes
- El sistema ahora soporta documentos de 200k+ caracteres
- Los documentos se procesan en lotes automáticamente
- Verificar logs para detalles del procesamiento

### Base de datos ChromaDB corrupta
```bash
# Eliminar y reiniciar ChromaDB
rm -rf chroma_db/
```

## 📝 Notas

- Los embeddings se almacenan en memoria con ChromaDB (por defecto)
- Para persistencia, configurar `chroma_db_dir` en [config.py](Challenge/config.py)
- El sistema usa cosine similarity para búsquedas
- El modelo multilingüe soporta español, inglés y otros idiomas

## 📄 Licencia

Este proyecto fue desarrollado como parte del Challenge de Get Talent.

## 👨‍💻 Autor

Martín - [Get Talent Challenge]

---

**Versión**: 1.0.0  
**Última actualización**: Diciembre 2025