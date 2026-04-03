import os
os.environ["IS_EMBEDDING_SERVER"] = "true"
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from backend.retrieval.embedder import LocalBGEHybridEncoder

app = FastAPI(title="Embedding Service", version="1.0.0")

# Singleton cho bộ nạp mô hình cục bộ bên trong container này
_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = LocalBGEHybridEncoder()
    return _encoder

class EmbedRequest(BaseModel):
    texts: Union[str, List[str]]
    batch_size: int = 16

@app.get("/health")
def health():
    return {"status": "ok", "onnx": get_encoder().is_onnx}

@app.post("/embed_dense")
async def embed_dense(request: EmbedRequest):
    try:
        vecs = get_encoder().encode_dense(request.texts, batch_size=request.batch_size)
        return {"vectors": vecs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed_sparse")
async def embed_sparse(request: EmbedRequest):
    try:
        # BGE-M3 (FlagEmbedding) trả về SparseVector Object (indices, values)
        # Chúng ta cần đảm bảo định dạng trả về tương thích với RemoteEmbedder sau này
        sparse_vecs = get_encoder().encode_sparse_documents(request.texts, batch_size=request.batch_size)
        return {
            "vectors": [
                {"indices": sv.indices, "values": sv.values} for sv in sparse_vecs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
