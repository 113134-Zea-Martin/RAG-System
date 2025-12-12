from typing import Dict, List, Optional
from datetime import datetime

class DocumentStorage:
    def __init__(self):
        self.documents: Dict[str, Dict] = {}
        self.embeddings: Dict[str, List[float]] = {}

    def save_document(self, title: str, content: str) -> str:
        document_id = f"{len(self.documents) + 1}"
        self.documents[document_id] = {
            "title": title,
            "content": content,
            "created_at": datetime.utcnow()
        }
        return document_id

    def get_document(self, document_id: str) -> Optional[Dict]:
        return self.documents.get(document_id)

    def create_embedding(self, document_id: str, content: str) -> None:
        embedding = [float(ord(c)) / 100.0 for c in content[:100]]
        self.embeddings[document_id] = embedding

    def search(self, query: str) -> List[Dict]:
        results = []
        for doc_id, doc in self.documents.items():
            if query.lower() in doc["content"].lower():
                results.append({
                    "document_id": doc_id,
                    "title": doc["title"],
                    "content_snippet": doc["content"][:100],
                    "similarity_score": 0.9
                })
        return results

    def answer_question(self, question: str) -> tuple[str, List[Dict]]:
        # Dummy question answering logic (replace with actual QA logic)
        context = []
        for doc_id, doc in self.documents.items():
            if question.lower() in doc["content"].lower():
                context.append({
                    "document_id": doc_id,
                    "content_snippet": doc["content"][:100],
                    "similarity_score": 0.8  # Dummy score
                })
        answer = "This is a dummy answer based on the documents."
        return answer, context