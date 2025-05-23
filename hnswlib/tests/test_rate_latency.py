import hnswlib
import asyncio
import requests
import numpy as np

# 不同rate下一直发请求的ann延迟
rate_list = [1, 5, 20, 80, 200, 500, 1000]

# curl请求并记录请求返回的耗时
async def send_query(rate, anns_search_range):
    api_url = 'http://localhost:8000/generate'
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=200)) as session:
            payload = {"queries": [query], "topk": topk, "return_scores": True}
            async with session.post(url=api_url, json=payload, headers={"Content-Type": "application/json"}) as response:
                res = await response.json()
                info = '\n'.join([r['document']['contents'] for r in res['result'][0]])
                return request_id, f"{ctx} <information> {info} </information>"