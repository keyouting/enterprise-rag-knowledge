from .base import LLMBasedAgent
from typing import Dict, Any, List


class GeneratorAgent(LLMBasedAgent):
    def __init__(self, name: str = "GeneratorAgent", model: str = "gpt-4"):
        super().__init__(name, model)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        retrieved = input_data.get("retrieved", [])
        
        self.log("基于检索结果生成答案...")
        
        context = self._build_context(retrieved)
        
        system_prompt = """你是一个企业知识库问答助手。请基于提供的参考资料回答问题。
如果参考资料中没有相关信息，请明确说明。
回答要准确、简洁、专业。"""
        
        prompt = f"""问题: {query}

参考资料:
{context}

请基于以上参考资料回答问题:"""
        
        if self.api_key:
            answer = self.call_llm(prompt, system_prompt)
        else:
            answer = self._simulate_answer(query, retrieved)
        
        self.add_message("assistant", f"生成答案 ({len(answer)} 字符)")
        
        return {
            "query": query,
            "answer": answer,
            "context_used": context,
            "sources": [r["filename"] for r in retrieved]
        }
    
    def _build_context(self, retrieved: List[Dict]) -> str:
        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            context_parts.append(f"[{i}] {doc['filename']} (相关度: {doc['relevance_score']}):\n{doc['content']}")
        return "\n\n".join(context_parts)
    
    def _simulate_answer(self, query: str, retrieved: List[Dict]) -> str:
        if not retrieved:
            return "抱歉，知识库中没有找到相关信息。"
        
        top_doc = retrieved[0]
        content = top_doc["content"]
        
        answer = f"根据 {top_doc['filename']} 的内容:\n\n"
        
        sentences = content.split('.')
        relevant = [s.strip() for s in sentences if any(word in s.lower() for word in query.lower().split())]
        
        if relevant:
            answer += '. '.join(relevant[:3]) + '。\n\n'
        else:
            answer += content[:300] + '...\n\n'
        
        answer += f"\n参考来源: {', '.join([r['filename'] for r in retrieved])}"
        
        return answer