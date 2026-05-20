from .base import BaseAgent
from typing import Dict, Any, List
import re


class RetrievalAgent(BaseAgent):
    def __init__(self, name: str = "RetrievalAgent"):
        super().__init__(name)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        chunks = input_data.get("chunks", [])
        embeddings = input_data.get("embeddings", [])
        top_k = input_data.get("top_k", 3)
        
        self.log(f"检索查询: {query[:50]}...")
        
        query_embedding = self._generate_query_embedding(query)
        
        similarities = []
        for i, chunk_embedding in enumerate(embeddings):
            sim = self._cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        retrieved = []
        for idx, score in top_results:
            chunk = chunks[idx]
            retrieved.append({
                "content": chunk["content"],
                "filename": chunk["filename"],
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "relevance_score": round(score, 4)
            })
        
        self.log(f"检索到 {len(retrieved)} 个相关文档块")
        
        return {
            "query": query,
            "retrieved": retrieved,
            "retrieval_count": len(retrieved)
        }
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        words = query.lower().split()
        word_freq = {}
        for word in words:
            word = re.sub(r'[^a-z\u4e00-\u9fff]', '', word)
            if word:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        embedding = [0.0] * 128
        for word, freq in word_freq.items():
            idx = hash(word) % 128
            embedding[idx] += freq * 0.1
        
        norm = sum(x**2 for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(x**2 for x in vec1) ** 0.5
        norm2 = sum(x**2 for x in vec2) ** 0.5
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0.0