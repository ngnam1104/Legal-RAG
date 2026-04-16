from backend.retrieval.chunker.core import AdvancedLegalChunker

# Expose a singleton instance matching the old chunker.py interface
chunker = AdvancedLegalChunker()
