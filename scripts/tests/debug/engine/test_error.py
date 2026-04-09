import asyncio
import traceback
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.agent.chat_engine import rag_engine

async def test():
    try:
        async for event in rag_engine.chat(
            session_id='test1234', 
            query='Quy định về thời gian phản hồi?', 
            mode='LEGAL_QA', 
            llm_preset='groq_8b', 
            top_k=3, 
            use_reflection=False, 
            use_rerank=False, 
            file_path=None
        ):
            print(event)
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
