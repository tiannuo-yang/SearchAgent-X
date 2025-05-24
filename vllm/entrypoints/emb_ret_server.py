import uvicorn
from fastapi import FastAPI, Request
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import JSONResponse, Response
from sentence_transformers import SentenceTransformer
import ssl
import os
import hnswlib
import numpy as np
import datasets
import asyncio
from functools import partial
import time
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, config_dir)
from config import (
    EMBEDDER,
    HNSW_INDEX,
    CORPUS
)

TIMEOUT_KEEP_ALIVE = 30  # seconds
MAX_WORKERS_EMB = 4
MAX_WORKERS_ANN = 4
# Initialize FastAPI application
app = FastAPI()
executor_emb = ThreadPoolExecutor(max_workers=MAX_WORKERS_EMB)
executor_ann = ThreadPoolExecutor(max_workers=MAX_WORKERS_ANN)
# executor_stall = ThreadPoolExecutor(max_workers=24)

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Load SentenceTransformer model
embedder = SentenceTransformer(EMBEDDER, device="cuda")

# Load HNSW index
index = hnswlib.Index(space='l2', dim=384)
index.load_index(HNSW_INDEX)
index.set_num_threads(6)

# Load corpus
corpus = datasets.load_dataset(
    'json',
    data_files=CORPUS,
    split="train",
)
sentences = corpus['contents']
print(f'Data loaded: {len(sentences)}')  # Expected length: 21015324
del corpus  # Free up memory

# Combined embedding and retrieval function (synchronous)
async def embed_and_retrieve(prompts, top_k, anns_search_range, req_id):
    """
    Combines embedding and retrieval operations, returning the nearest neighbor texts.
    
    Args:
        prompts (list[str]): List of input texts.
        top_k (int): Number of nearest neighbors to return.
        anns_search_range (int): Search range (-1 for brute-force search, >=0 for approximate search).
    
    Returns:
        list[list[str]]: List of nearest neighbor texts for each input.
    """
    t0 = time.time()
    # Step 1: Generate embeddings
    loop1 = asyncio.get_event_loop()
    encode_func = partial(embedder.encode, sentences=prompts, normalize_embeddings=True, convert_to_numpy=True, )
    embeds = await loop1.run_in_executor(executor_emb, encode_func)

    t1 = time.time()
    # Step 2: Perform retrieval
    embeddings = np.array(embeds, dtype=np.float32)
    
    top_k = int(top_k)
    req_id = int(req_id)
    anns_search_range = int(anns_search_range)
    
    loop2 = asyncio.get_event_loop()
    if anns_search_range >= 0:
        index.set_ef(anns_search_range)
        neighbors, distances, is_stop_by_stall = await loop2.run_in_executor(executor_ann, index.knn_query_with_stall, embeddings, top_k, req_id)
    else:
        neighbors, distances = await loop2.run_in_executor(executor_ann, index.bf_knn_query, embeddings, top_k)
        is_stop_by_stall = False
    
    neighbors = np.asarray(neighbors, dtype=np.int32)
    neighbor_texts = [[sentences[idx] for idx in row] for row in neighbors]
    t2 = time.time()

    # Step 3: Return current load status
    active_threads = len([t for t in executor_ann._threads if t.is_alive()])
    queued_tasks = executor_ann._work_queue.qsize()
    t_embed = t1 - t0
    t_ann = t2 - t1
    response = {"neighbor_texts": neighbor_texts, "active_threads": active_threads, "queued_tasks": queued_tasks, "is_stop_by_stall": is_stop_by_stall, "t_embed": t_embed, "t_ann": t_ann}
    return response

# Health check endpoint
@app.get("/health")
def health() -> Response:
    """Health check endpoint"""
    return Response(status_code=200)

@app.get("/reset")
def reset() -> Response:
    global executor_emb, executor_ann
    if executor_emb:
        executor_emb.shutdown(wait=True, cancel_futures=True)
    if executor_ann:
        executor_ann.shutdown(wait=True, cancel_futures=True)
    executor_emb = ThreadPoolExecutor(max_workers=MAX_WORKERS_EMB)
    executor_ann = ThreadPoolExecutor(max_workers=MAX_WORKERS_ANN)
    return Response(status_code=200)

@app.get("/check_stall")
async def check_stall(step_low_ema: float = 0.90) -> Response:
    # Returns the IDs of stalled requests
    loop = asyncio.get_event_loop()
    stopeed_requests = await loop.run_in_executor(executor_emb, index.early_stop, step_low_ema)
    return JSONResponse({"stopeed_requests": stopeed_requests.tolist()})

# Non-blocking embedding and retrieval endpoint
@app.post("/embed_and_retrieve")
async def process_request(request: Request) -> Response:
    """
    Processes embedding and retrieval requests, non-blocking version.
    
    Request example:
    {
        "prompts": ["What is AI?", "How does retrieval work?"],
        "top_k": 5,
        "anns_search_range": 100  // Optional, defaults to -1 (brute-force search)
    }
    
    Returns:
        JSON formatted nearest neighbor texts.
    """
    request_dict = await request.json()
    prompts = request_dict["prompts"]  # Expected to be a list of strings
    top_k = request_dict["top_k"]
    anns_search_range = request_dict.get("anns_search_range", -1)  # Default to brute-force search
    req_id = request_dict.get("request_id", -1) 
    
    t3 = time.time()

    # Offload the combined task to a single thread
    response = await embed_and_retrieve(prompts, top_k, anns_search_range, req_id)
    t4 = time.time()
    response["t_thread"] = t4 - t3
    return JSONResponse(response)


# Run the application
if __name__ == '__main__':
    app.root_path = None
    uvicorn.run(
        app,
        host=None,  # Listen on all interfaces
        port=8500,      # Run on port 8000
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_cert_reqs=ssl.CERT_NONE,
        log_level='debug',
        http='h11',         # Alternative to default 'httptools' (more stable)
        loop='uvloop'       # Use high-performance event loop
    )