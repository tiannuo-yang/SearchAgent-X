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
# 初始化 FastAPI 应用
app = FastAPI()
executor_emb = ThreadPoolExecutor(max_workers=MAX_WORKERS_EMB)
executor_ann = ThreadPoolExecutor(max_workers=MAX_WORKERS_ANN)
# executor_stall = ThreadPoolExecutor(max_workers=24)

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 加载 SentenceTransformer 模型
embedder = SentenceTransformer(EMBEDDER, device="cuda")

# 加载 HNSW 索引
index = hnswlib.Index(space='l2', dim=384)
index.load_index(HNSW_INDEX)
index.set_num_threads(6)

# 加载语料库
corpus = datasets.load_dataset(
    'json',
    data_files=CORPUS,
    split="train",
)
sentences = corpus['contents']
print(f'数据加载完成: {len(sentences)}')  # 预期长度: 21015324
del corpus  # 释放内存

# 合并的嵌入和检索函数（同步）
async def embed_and_retrieve(prompts, top_k, anns_search_range, req_id):
    """
    合并嵌入和检索操作，返回最近邻文本。
    
    Args:
        prompts (list[str]): 输入的文本列表。
        top_k (int): 返回的最近邻数量。
        anns_search_range (int): 搜索范围（-1 表示暴力搜索，>=0 表示近似搜索）。
    
    Returns:
        list[list[str]]: 每个输入的最近邻文本列表。
    """
    t0 = time.time()
    # Step 1: 生成嵌入
    loop1 = asyncio.get_event_loop()
    encode_func = partial(embedder.encode, sentences=prompts, normalize_embeddings=True, convert_to_numpy=True, )
    embeds = await loop1.run_in_executor(executor_emb, encode_func)

    t1 = time.time()
    # Step 2: 执行检索
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

    # Step 3: 返回当前负载情况
    active_threads = len([t for t in executor_ann._threads if t.is_alive()])
    queued_tasks = executor_ann._work_queue.qsize()
    t_embed = t1 - t0
    t_ann = t2 - t1
    response = {"neighbor_texts": neighbor_texts, "active_threads": active_threads, "queued_tasks": queued_tasks, "is_stop_by_stall": is_stop_by_stall, "t_embed": t_embed, "t_ann": t_ann}
    return response

# 健康检查端点
@app.get("/health")
def health() -> Response:
    """健康检查端点"""
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
    # 返回被暂停的请求的id
    loop = asyncio.get_event_loop()
    stopeed_requests = await loop.run_in_executor(executor_emb, index.early_stop, step_low_ema)
    return JSONResponse({"stopeed_requests": stopeed_requests.tolist()})

# 非阻塞的嵌入和检索端点
@app.post("/embed_and_retrieve")
async def process_request(request: Request) -> Response:
    """
    处理嵌入和检索请求，非阻塞版本。
    
    请求示例:
    {
        "prompts": ["什么是AI?", "检索如何工作?"],
        "top_k": 5,
        "anns_search_range": 100  // 可选，默认为 -1（暴力搜索）
    }
    
    返回:
        JSON 格式的邻居文本。
    """
    request_dict = await request.json()
    prompts = request_dict["prompts"]  # 预期为字符串列表
    top_k = request_dict["top_k"]
    anns_search_range = request_dict.get("anns_search_range", -1)  # 默认暴力搜索
    req_id = request_dict.get("request_id", -1) 
    
    t3 = time.time()

    # 将合并的任务卸载到单个线程
    response = await embed_and_retrieve(prompts, top_k, anns_search_range, req_id)
    t4 = time.time()
    response["t_thread"] = t4 - t3
    return JSONResponse(response)


# 运行应用
if __name__ == '__main__':
    app.root_path = None
    uvicorn.run(
        app,
        host=None,  # 监听所有接口
        port=8500,       # 运行在 8000 端口
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_cert_reqs=ssl.CERT_NONE,
        log_level='debug',
        http='h11',         # 替代默认的 'httptools'（更稳定）
        loop='uvloop'       # 使用高性能事件循环
    )