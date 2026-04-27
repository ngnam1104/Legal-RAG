"""
SubTimer: Tiện ích đo thời gian chi tiết từng bước nhỏ trong pipeline RAG.

Sử dụng:
    timer = SubTimer("Retrieve")
    with timer.step("Qdrant_Search"):
        ... # code tìm kiếm
    with timer.step("Rerank"):
        ... # code rerank
    
    # Lấy kết quả:
    timer.results()  
    # => {"Retrieve.Qdrant_Search": 0.45, "Retrieve.Rerank": 0.19}
"""
import time
from contextlib import contextmanager
from typing import Dict


class SubTimer:
    """Đo thời gian chi tiết của các sub-steps bên trong một node LangGraph."""

    def __init__(self, parent_name: str):
        """
        Args:
            parent_name: Tên node cha (VD: "Retrieve", "Generate", "Route").
                         Sẽ được dùng làm prefix cho tên sub-step.
        """
        self._parent = parent_name
        self._timings: Dict[str, float] = {}

    @contextmanager
    def step(self, name: str):
        """Context manager đo 1 sub-step.
        
        Args:
            name: Tên bước nhỏ (VD: "Qdrant_Search", "LLM_Call").
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - t0
            key = f"{self._parent}.{name}"
            # Cộng dồn nếu cùng tên (VD: nhiều lần gọi Neo4j)
            self._timings[key] = self._timings.get(key, 0.0) + duration

    def results(self) -> Dict[str, float]:
        """Trả về dict {sub_step_name: duration_seconds}."""
        return dict(self._timings)
