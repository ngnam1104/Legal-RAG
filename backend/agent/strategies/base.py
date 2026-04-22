from abc import ABC, abstractmethod
from typing import Dict, Any
from backend.agent.state import AgentState

class BaseRAGStrategy(ABC):
    @abstractmethod
    def understand(self, state: AgentState) -> Dict[str, Any]:
        pass

    @abstractmethod
    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate(self, state: AgentState) -> Dict[str, Any]:
        pass
