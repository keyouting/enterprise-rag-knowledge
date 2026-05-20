from agents.base import BaseAgent
from agents.document_agent import DocumentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.generator_agent import GeneratorAgent
from agents.verifier_agent import VerifierAgent
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from rich.console import Console


@dataclass
class ChainStep:
    name: str
    agent: BaseAgent
    input_key: str
    output_key: str
    description: str


class RAGChain:
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.steps: List[ChainStep] = []
        self.results: Dict[str, Any] = {}
        self.console = Console()
        
        self._init_chain()
    
    def _init_chain(self):
        doc_agent = DocumentAgent()
        retrieval_agent = RetrievalAgent()
        generator_agent = GeneratorAgent(self.model)
        verifier_agent = VerifierAgent()
        
        self.steps = [
            ChainStep("Document", doc_agent, "docs_path", "documents", "加载并处理文档"),
            ChainStep("Retrieval", retrieval_agent, "query", "retrieved", "检索相关文档"),
            ChainStep("Generator", generator_agent, "context", "answer", "生成答案"),
            ChainStep("Verifier", verifier_agent, "answer", "verification", "验证答案准确性")
        ]
    
    def run(self, query: str, docs_path: str = "sample_docs") -> Dict[str, Any]:
        self.console.print("[bold cyan]=== 开始RAG长链推理流程 ===[/bold cyan]\n")
        
        start_time = datetime.now()
        
        self.results = {"query": query, "docs_path": docs_path}
        
        for step in self.steps:
            self.console.print(f"[bold yellow]Step {self.steps.index(step)+1}: {step.name}[/bold yellow] - {step.description}")
            
            if step.name == "Document":
                input_data = {"docs_path": docs_path}
            elif step.name == "Retrieval":
                docs_result = self.results.get("documents", {})
                input_data = {
                    "query": query,
                    "chunks": docs_result.get("chunks", []),
                    "embeddings": docs_result.get("embeddings", []),
                    "top_k": 3
                }
            elif step.name == "Generator":
                input_data = {
                    "query": query,
                    "retrieved": self.results.get("retrieved", {}).get("retrieved", [])
                }
            elif step.name == "Verifier":
                gen_result = self.results.get("answer", {})
                input_data = {
                    "query": query,
                    "answer": gen_result.get("answer", ""),
                    "retrieved": self.results.get("retrieved", {}).get("retrieved", [])
                }
            else:
                input_data = {}
            
            result = step.agent.process(input_data)
            self.results[step.output_key] = result
            
            self._log_step_result(step.name, result)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        self.console.print(f"\n[bold green]RAG流程执行完成! 耗时: {elapsed:.2f}s[/bold green]")
        
        return self.results
    
    def _log_step_result(self, step_name: str, result: Dict):
        if step_name == "Document":
            count = result.get("chunk_count", 0)
            self.console.print(f"  -> 处理 {count} 个文本块")
        elif step_name == "Retrieval":
            count = result.get("retrieval_count", 0)
            self.console.print(f"  -> 检索到 {count} 个相关文档")
        elif step_name == "Generator":
            ans_len = len(result.get("answer", ""))
            self.console.print(f"  -> 生成答案 ({ans_len} 字符)")
        elif step_name == "Verifier":
            ver = result.get("verification", {})
            conf = ver.get("confidence", 0)
            self.console.print(f"  -> 验证完成, 置信度: {conf}%")
    
    def get_final_answer(self) -> str:
        gen = self.results.get("answer", {})
        ver = self.results.get("verification", {})
        answer = gen.get("answer", "无答案")
        confidence = ver.get("verification", {}).get("confidence", 0)
        return f"{answer}\n\n[置信度: {confidence}%]"