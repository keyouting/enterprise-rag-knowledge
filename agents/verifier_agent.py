from .base import BaseAgent
from typing import Dict, Any, List


class VerifierAgent(BaseAgent):
    def __init__(self, name: str = "VerifierAgent"):
        super().__init__(name)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        answer = input_data.get("answer", "")
        retrieved = input_data.get("retrieved", [])
        
        self.log("验证答案准确性...")
        
        verification = self._verify_answer(query, answer, retrieved)
        
        self.add_message("assistant", f"验证完成, 置信度: {verification['confidence']}%")
        
        return {
            "query": query,
            "answer": answer,
            "verification": verification
        }
    
    def _verify_answer(self, query: str, answer: str, retrieved: List[Dict]) -> Dict[str, Any]:
        confidence = self._calculate_confidence(answer, retrieved)
        
        issues = []
        if len(answer) < 20:
            issues.append("答案过短，可能不完整")
        
        if "不知道" in answer or "不确定" in answer:
            issues.append("答案表示不确定")
        
        if not retrieved:
            issues.append("无参考资料支撑")
        
        return {
            "confidence": confidence,
            "issues": issues,
            "is_reliable": confidence >= 70,
            "sources_count": len(retrieved)
        }
    
    def _calculate_confidence(self, answer: str, retrieved: List[Dict]) -> int:
        if not retrieved:
            return 0
        
        base_score = 60
        
        if len(retrieved) >= 2:
            base_score += 10
        
        if len(answer) > 100:
            base_score += 10
        
        if len(answer) > 200:
            base_score += 10
        
        relevance_scores = [r.get("relevance_score", 0) for r in retrieved]
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            base_score += int(avg_relevance * 10)
        
        return min(100, base_score)