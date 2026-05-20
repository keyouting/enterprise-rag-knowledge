import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from rich.console import Console

console = Console()


@dataclass
class AgentMessage:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    def __init__(self, name: str, model: str = "gpt-4"):
        self.name = name
        self.model = model
        self.messages: List[AgentMessage] = []
        self.console = Console()

    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        pass

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        msg = AgentMessage(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        return msg

    def get_messages(self) -> List[AgentMessage]:
        return self.messages

    def clear_history(self):
        self.messages.clear()

    def log(self, message: str, style: str = "bold green"):
        self.console.print(f"[{style}]{self.name}:[/{style}] {message}")


class LLMBasedAgent(BaseAgent):
    def __init__(self, name: str, model: str = "gpt-4", api_key: Optional[str] = None):
        super().__init__(name, model)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content