"""
Search Intent / Keyword Extraction — icllmlib Adapter
=====================================================
Sử dụng thư viện nội bộ ``icllmlib`` để trích xuất từ khóa từ câu query.

* ``extract_keywords_from_query``  — sync  (blocking I/O)
* ``aextract_keywords_from_query`` — async  (``asyncio.to_thread``)

Server: http://10.9.3.241:5564/api/Qas/v2  (model llm3.1-sea)
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cấu hình icllmlib — Khởi tạo 1 lần (module-level singleton)
# ---------------------------------------------------------------------------
_llm_client = None


def _get_llm_client():
    """Lazy-init singleton cho icllmlib client."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    try:
        from icllmlib import LLM

        _llm_client = LLM(
            model="llm3.1-sea",
            url="http://10.9.3.241:5564/api/Qas/v2",
        )
        logger.info(
            "✅ [utils_intent] icllmlib LLM client khởi tạo — "
            "model=llm3.1-sea, url=http://10.9.3.241:5564/api/Qas/v2"
        )
    except ImportError:
        logger.error(
            "❌ [utils_intent] Không tìm thấy thư viện icllmlib. "
            "Hãy cài đặt: pip install icllmlib"
        )
        _llm_client = None
    except Exception:
        logger.exception("❌ [utils_intent] Lỗi khi khởi tạo icllmlib LLM client.")
        _llm_client = None

    return _llm_client


# ---------------------------------------------------------------------------
# Sync API
# ---------------------------------------------------------------------------
def extract_keywords_from_query(query: str) -> List[str]:
    """
    Trích xuất từ khóa tìm kiếm từ câu hỏi người dùng (blocking I/O).

    Args:
        query: Câu hỏi / truy vấn gốc.

    Returns:
        Danh sách từ khóa.  Trả ``[]`` nếu có lỗi.
    """
    if not query or not query.strip():
        return []

    client = _get_llm_client()
    if client is None:
        logger.warning("⚠️ [utils_intent] icllmlib client chưa sẵn sàng — trả [].")
        return []

    try:
        keywords: List[str] = client.get_keywords(query)
        if not isinstance(keywords, list):
            keywords = [str(keywords)] if keywords else []
        logger.debug(
            "🔑 [utils_intent] Query: %r → Keywords: %s",
            query, keywords,
        )
        return keywords
    except Exception:
        logger.exception(
            "❌ [utils_intent] Lỗi khi gọi get_keywords cho query: %r", query,
        )
        return []


# ---------------------------------------------------------------------------
# Async API — wrap sync qua asyncio.to_thread (Python 3.9+)
# ---------------------------------------------------------------------------
async def aextract_keywords_from_query(query: str) -> List[str]:
    """
    Phiên bản async của ``extract_keywords_from_query``.
    Sử dụng ``asyncio.to_thread`` để không block event-loop của FastAPI.
    """
    return await asyncio.to_thread(extract_keywords_from_query, query)
