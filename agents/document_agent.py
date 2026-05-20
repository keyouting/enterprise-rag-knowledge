from .base import BaseAgent
from typing import Dict, Any, List
import os
import re
import hashlib


class DocumentAgent(BaseAgent):
    def __init__(self, name: str = "DocumentAgent"):
        super().__init__(name)
        self.documents: List[Dict[str, Any]] = []
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        docs_path = input_data.get("docs_path", "sample_docs")
        chunk_size = input_data.get("chunk_size", 500)
        chunk_overlap = input_data.get("chunk_overlap", 50)
        
        self.log(f"加载文档目录: {docs_path}")
        
        self.documents = self._load_documents(docs_path)
        self.log(f"加载 {len(self.documents)} 个文档")
        
        chunks = self._chunk_documents(chunk_size, chunk_overlap)
        self.log(f"切分为 {len(chunks)} 个文本块")
        
        embeddings = self._generate_embeddings(chunks)
        self.log("生成向量嵌入完成")
        
        return {
            "documents": self.documents,
            "chunks": chunks,
            "embeddings": embeddings,
            "chunk_count": len(chunks)
        }
    
    def _load_documents(self, docs_path: str) -> List[Dict[str, Any]]:
        documents = []
        if not os.path.exists(docs_path):
            return documents
        
        for filename in os.listdir(docs_path):
            filepath = os.path.join(docs_path, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append({
                        "filename": filename,
                        "content": content,
                        "size": len(content),
                        "doc_id": hashlib.md5(content.encode()).hexdigest()[:8]
                    })
                except Exception as e:
                    self.log(f"加载失败 {filename}: {e}", "bold red")
        
        return documents
    
    def _chunk_documents(self, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        chunks = []
        for doc in self.documents:
            text = doc["content"]
            sentences = re.split(r'[.!?。！？\n]+', text)
            
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_chunk) + len(sentence) > chunk_size:
                    if current_chunk:
                        chunks.append({
                            "content": current_chunk,
                            "doc_id": doc["doc_id"],
                            "filename": doc["filename"],
                            "chunk_id": hashlib.md5(current_chunk.encode()).hexdigest()[:8]
                        })
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence
            
            if current_chunk:
                chunks.append({
                    "content": current_chunk,
                    "doc_id": doc["doc_id"],
                    "filename": doc["filename"],
                    "chunk_id": hashlib.md5(current_chunk.encode()).hexdigest()[:8]
                })
        
        return chunks
    
    def _generate_embeddings(self, chunks: List[Dict]) -> List[List[float]]:
        embeddings = []
        for chunk in chunks:
            embedding = self._simple_embedding(chunk["content"])
            embeddings.append(embedding)
        return embeddings
    
    def _simple_embedding(self, text: str) -> List[float]:
        words = text.lower().split()
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