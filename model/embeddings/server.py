"""
Local Embedding Service using Google Gemma Embedding Model
Provides a FastAPI server for local inference to reduce cloud API costs.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from functools import lru_cache
import hashlib
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Try different inference backends in order of preference
try:
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers import AutoTokenizer, AutoModel
    BACKEND = "sentence_transformers"  # Preferred for embeddings
except ImportError:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        BACKEND = "transformers"
    except ImportError:
        try:
            import onnxruntime as ort
            BACKEND = "onnx"
        except ImportError:
            BACKEND = None

# Request/Response Models
class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed")
    model: Optional[str] = Field(default="gemma", description="Model to use")
    
class BatchEmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="Texts to embed")
    model: Optional[str] = Field(default="gemma", description="Model to use")

class EmbedResponse(BaseModel):
    embedding: List[float]
    model: str
    dimensions: int
    processing_time_ms: float

class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int
    processing_time_ms: float
    batch_size: int

class HealthResponse(BaseModel):
    status: str
    backend: str
    model_loaded: bool
    cache_size: int
    uptime_seconds: float

# Global model cache and stats
model_cache = {}
embedding_cache = {}
start_time = time.time()
request_count = 0
total_processing_time = 0.0

app = FastAPI(
    title="CrescendAI Local Embedding Service",
    description="Local Gemma embedding inference for cost optimization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "https://*.crescend.ai"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def get_cache_key(text: str, model: str) -> str:
    """Generate cache key for text/model combination."""
    content = f"{model}:{text}"
    return hashlib.md5(content.encode()).hexdigest()

@lru_cache(maxsize=1)
def load_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load the embedding model with caching."""
    global model_cache
    
    if model_name in model_cache:
        return model_cache[model_name]
    
    print(f"Loading model {model_name} with backend {BACKEND}...")
    
    if BACKEND == "sentence_transformers":
        try:
            # Use sentence-transformers for optimized embedding inference
            model = SentenceTransformer(model_name)
            
            # Try to use MPS on Apple Silicon, or CUDA if available
            if torch.backends.mps.is_available():
                model = model.to("mps")
                print("Using MPS (Apple Metal) acceleration")
            elif torch.cuda.is_available():
                model = model.to("cuda")
                print("Using CUDA acceleration")
            else:
                print("Using CPU inference")
            
            model_cache[model_name] = {"model": model}
            return model_cache[model_name]
            
        except Exception as e:
            print(f"Failed to load with sentence-transformers: {e}")
            # Fall back to transformers
            pass
    
    # Fallback to raw transformers
    if BACKEND in ["sentence_transformers", "transformers"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
            
            # Use MPS if available on Mac
            if torch.backends.mps.is_available():
                model = model.to("mps")
                print("Using MPS (Apple Metal) acceleration")
            elif torch.cuda.is_available():
                model = model.to("cuda")
                print("Using CUDA acceleration")
            else:
                print("Using CPU inference")
            
            model.eval()
            model_cache[model_name] = {"tokenizer": tokenizer, "model": model}
            return model_cache[model_name]
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    else:
        raise RuntimeError(f"No suitable backend available. Install: uv add torch transformers sentence-transformers")

def compute_embedding(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    """Compute embedding for a single text."""
    cache_key = get_cache_key(text, model_name)
    
    # Check cache first
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    # Load model
    model_components = load_model(model_name)
    model = model_components["model"]
    
    start_time = time.time()
    
    if BACKEND == "sentence_transformers" and isinstance(model, SentenceTransformer):
        # Use sentence-transformers (optimized for embeddings)
        try:
            embeddings = model.encode([text], convert_to_tensor=True, normalize_embeddings=True)
            embedding_list = embeddings[0].cpu().numpy().tolist()
        except Exception as e:
            print(f"Sentence-transformers failed: {e}, falling back to manual method")
            # Fall back to manual method
            model_components = load_model(model_name)  # This will fall back to transformers
            return compute_embedding_manual(text, model_components)
    
    elif "tokenizer" in model_components:
        # Use raw transformers with manual pooling
        return compute_embedding_manual(text, model_components)
    
    else:
        raise RuntimeError("No suitable embedding backend available")
    
    # Cache result
    embedding_cache[cache_key] = embedding_list
    
    # Limit cache size
    if len(embedding_cache) > 10000:
        # Remove oldest 20% of entries
        keys_to_remove = list(embedding_cache.keys())[:2000]
        for key in keys_to_remove:
            del embedding_cache[key]
    
    processing_time = (time.time() - start_time) * 1000
    print(f"Computed embedding in {processing_time:.1f}ms (dim={len(embedding_list)})")
    
    return embedding_list

def compute_embedding_manual(text: str, model_components: dict) -> List[float]:
    """Manual embedding computation using transformers."""
    tokenizer = model_components["tokenizer"]
    model = model_components["model"]
    
    with torch.no_grad():
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        outputs = model(**inputs)
        
        # Pool embeddings (mean pooling over sequence length)
        embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        
        # Mean pooling with attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        embeddings = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to list
        return embeddings.cpu().numpy()[0].tolist()

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest, background_tasks: BackgroundTasks):
    """Embed a single text."""
    global request_count, total_processing_time
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    start_time = time.time()
    
    try:
        embedding = compute_embedding(request.text, request.model or "sentence-transformers/all-MiniLM-L6-v2")
        processing_time = (time.time() - start_time) * 1000
        
        # Update stats
        background_tasks.add_task(update_stats, processing_time)
        
        return EmbedResponse(
            embedding=embedding,
            model=request.model or "gemma",
            dimensions=len(embedding),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.post("/embed/batch", response_model=BatchEmbedResponse) 
async def embed_batch(request: BatchEmbedRequest, background_tasks: BackgroundTasks):
    """Embed multiple texts in batch for efficiency."""
    global request_count, total_processing_time
    
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    start_time = time.time()
    
    try:
        embeddings = []
        for text in request.texts:
            if text.strip():  # Skip empty texts
                embedding = compute_embedding(text, request.model or "sentence-transformers/all-MiniLM-L6-v2")
                embeddings.append(embedding)
            else:
                # Return zero vector for empty text
                embeddings.append([0.0] * (len(embeddings[0]) if embeddings else 768))
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update stats
        background_tasks.add_task(update_stats, processing_time, len(request.texts))
        
        return BatchEmbedResponse(
            embeddings=embeddings,
            model=request.model or "gemma",
            dimensions=len(embeddings[0]) if embeddings else 0,
            processing_time_ms=processing_time,
            batch_size=len(embeddings)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch embedding failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global start_time, request_count, total_processing_time
    
    uptime = time.time() - start_time
    model_loaded = len(model_cache) > 0
    
    return HealthResponse(
        status="healthy" if BACKEND else "no_backend",
        backend=BACKEND or "none",
        model_loaded=model_loaded,
        cache_size=len(embedding_cache),
        uptime_seconds=uptime
    )

@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    global start_time, request_count, total_processing_time
    
    uptime = time.time() - start_time
    avg_processing_time = total_processing_time / max(request_count, 1)
    
    return {
        "uptime_seconds": uptime,
        "total_requests": request_count,
        "average_processing_time_ms": avg_processing_time,
        "cache_size": len(embedding_cache),
        "cache_hit_ratio": "N/A",  # TODO: Track cache hits
        "backend": BACKEND,
        "models_loaded": list(model_cache.keys())
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear embedding cache."""
    global embedding_cache
    cache_size = len(embedding_cache)
    embedding_cache.clear()
    return {"message": f"Cleared {cache_size} cached embeddings"}

def update_stats(processing_time: float, batch_size: int = 1):
    """Update global statistics."""
    global request_count, total_processing_time
    request_count += batch_size
    total_processing_time += processing_time

if __name__ == "__main__":
    print(f"Starting CrescendAI Embedding Service with backend: {BACKEND}")
    print("Available endpoints:")
    print("  POST /embed - Single text embedding")
    print("  POST /embed/batch - Batch text embedding")
    print("  GET /health - Health check")
    print("  GET /stats - Service statistics")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,  # Disable reload for production
        access_log=True
    )