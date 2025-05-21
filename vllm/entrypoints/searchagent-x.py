from typing import List, Optional, Union, AsyncGenerator, Dict, Tuple
from collections import deque
from collections import Counter as CollectionsCounter
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.core.policy import PolicyFactory
from concurrent.futures import ThreadPoolExecutor
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter
import numpy as np
import asyncio
import time
import aiohttp
import requests
import sys
import os
import gc
import json
import torch
import logging
from qadata import QAData
from exp import Exp_Config, Exp_Output
from exp import (
    OUTPUT_FILE,
    MODEL_LIST,
    DATA_LIST,
    PROMPT_TEMPLATE,
    SEARCH_LABLE_A,
    SEARCH_LABLE_B,
    INFORMATION_LABLE_A,
    INFORMATION_LABLE_B,
    IS_INSTRUCT,
    REQUEST_RATE,
    MAX_PROMPT_NUM,
    TOPK,
)

np.random.seed(1206)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_FILE),
        logging.StreamHandler()
    ]
)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class LLM:
    def __init__(self, exp_config: Exp_Config) -> None:
        """初始化LLM引擎和相关状态，使用Exp_Config实例"""
        self.exp_config = exp_config
        engine_args = EngineArgs(
            model=exp_config.model,
            enforce_eager=exp_config.enforce_eager,
            tensor_parallel_size=exp_config.tensor_parallel_size,
            max_model_len=exp_config.max_model_len,
            seed=exp_config.seed,
            enable_prefix_caching=exp_config.enable_prefix_cache,
            disable_log_stats=True,
            scheduler_delay_factor=exp_config.delay_factor,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args, usage_context=UsageContext.LLM_CLASS)
        self.request_counter = Counter()

    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        """获取分词器"""
        return self.llm_engine.tokenizer.tokenizer

    def _add_request(self, prompt: Optional[str], question: Optional[str] = None, max_possi_prefix_block_len: Optional[int] = 0, arr_time: Optional[float] = None, submit_time: Optional[float] = None) -> str:
        """添加请求到引擎"""
        request_id = str(next(self.request_counter))
        sampling_params = SamplingParams(
            max_tokens=self.exp_config.max_tokens,
            temperature=0,
            seed=self.exp_config.seed
        )
        if not arr_time:
            arr_time = time.time()

        self.requests_arr_time[request_id] = arr_time
        if question:
            self.request_questions[request_id] = question
            self.successive_requests.append([request_id])
            submit_time = arr_time
        else:
            self.prefix_record[request_id] = [max_possi_prefix_block_len]

        assert submit_time is not None, "Submit time is None"
        self.llm_engine.add_request(request_id=request_id, prompt=prompt, sampling_params=sampling_params, arrival_time=arr_time, submit_time=submit_time, max_possi_prefix_block_len=max_possi_prefix_block_len)

        return request_id
     
    def _priority_schedule(self,) -> None:
        # self.llm_engine.scheduler.skip_vllm_policy = True
        if len(self.llm_engine.scheduler.waiting) <= 1:
            return 
        
        if self.exp_config.priority_schedule == -1:
            now = time.time()
            seq_groups = self.llm_engine.scheduler.waiting
            req_ids = [seq_group.request_id for seq_group in seq_groups]
            wait_durations_cur = [now - self.requests_arr_time[r] for r in req_ids]
            self.llm_engine.scheduler.waiting = deque(elem for _, elem in sorted(zip(wait_durations_cur, seq_groups), key=lambda x: x[0], reverse=True))

        if self.exp_config.priority_schedule == 0:
            seq_groups = self.llm_engine.scheduler.waiting
            req_ids = [seq_group.request_id for seq_group in seq_groups]
            roots = [Exp_Output.find_root(self.successive_requests, r) for r in req_ids]
            search_counts = [len(r) for r in roots]

            self.llm_engine.scheduler.waiting = deque(elem for _, elem in sorted(zip(search_counts, seq_groups), key=lambda x: x[0], reverse=True))

        elif self.exp_config.priority_schedule >= 1:
            granularity = self.exp_config.priority_schedule

            seq_groups = self.llm_engine.scheduler.waiting
            req_ids = [seq_group.request_id for seq_group in seq_groups]
            roots = [Exp_Output.find_root(self.successive_requests, r) for r in req_ids]
            now = time.time()

            search_counts = [len(r) for r in roots]
            wait_durations = [now - self.requests_arr_time[r[0]] for r in roots]
            wait_durations_cur = [now - self.requests_arr_time[r] for r in req_ids]
            ctx_lengths = [sq.get_seqs()[0].data.get_len() for sq in seq_groups]

            # levels = [i/granularity * 100 for i in range(1, granularity + 1)]
            level_reqs = {idx: [] for idx in range(granularity)} # {0: [], 1: [], 2: [], 3: []}
            sc_levels = [(max(search_counts)-min(search_counts))*i/granularity+min(search_counts) for i in range(granularity)] # e.g., [3.0, 6.25, 9.5, 12.75] for search_counts = [3, 4, 5, 6, 7, 8, 10, 14, 15, 16]
            wd_levels = [(max(wait_durations)-min(wait_durations))*i/granularity+min(wait_durations) for i in range(granularity)]
            cl_levels = [(max(ctx_lengths)-min(ctx_lengths))*i/granularity+min(ctx_lengths) for i in range(granularity)]

            for i, r in enumerate(req_ids):
                for idx in range(granularity-1, -1, -1):  # for different levels, e.g., [3,2,1,0]
                    if (search_counts[i] > sc_levels[idx] or wait_durations[i] > wd_levels[idx] or ctx_lengths[i] > cl_levels[idx]) or (idx == 0):
                    # if search_counts[i] >= sc_levels[idx] or ctx_lengths[i] >= cl_levels[idx]:
                        # 按照 wait_durations_cur[i] 从大到小排序插入
                        insert_position = 0
                        while insert_position < len(level_reqs[idx]) and wait_durations_cur[i] <= wait_durations_cur[level_reqs[idx][insert_position]]:
                            insert_position += 1
                        level_reqs[idx].insert(insert_position, i)
                        break

            final_order = [r for idx in range(granularity-1, -1, -1) for r in level_reqs[idx]]
            temp = list(seq_groups)
            seq_groups.clear()
            seq_groups.extend([temp[i] for i in final_order])

    def _record_for_one_step(self, scheduler_outputs) -> None:
        """记录单步的token生成时间"""
        if not scheduler_outputs.is_empty():
            now = time.time()
            for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
                seq_group = scheduled_seq_group.seq_group
                seq_id = seq_group.request_id
                token_time = seq_group.get_last_latency(now)
                if seq_id in self.request_questions.keys(): # its a root seq
                    if scheduler_outputs.prompt_run:
                        self.requests_token_time[seq_id] = [[token_time]]
                    else:
                        self.requests_token_time[seq_id][0].append(token_time)
                else: # its a resumed seq
                    root_idx = next(i for i, row in enumerate(self.successive_requests) if seq_id in row)
                    root_seq = self.successive_requests[root_idx][0]
                    pre_seq = self.successive_requests[root_idx][-2]
                    if scheduler_outputs.prompt_run:
                        offset = (self.requests_arr_time[seq_id] -
                                  self.requests_arr_time[pre_seq] -
                                  sum(self.requests_token_time[root_seq][-1]))
                        self.requests_token_time[root_seq] += [offset, [token_time]]

                        assert seq_id in self.prefix_record.keys(), "Resumed seq not recorded in max possible prefix block len"
                        assert isinstance(scheduled_seq_group.hit_count, int), f"Hit count {scheduled_seq_group.hit_count} is not int"
                        self.prefix_record[seq_id].append(scheduled_seq_group.hit_count)
                        # self.prefix_monitor()
                    else:
                        self.requests_token_time[root_seq][-1].append(token_time)
                    

    async def run_engine_with_search(self, arrival_requests) -> Exp_Output:
        """运行引擎并处理搜索任务"""
        self.outputs: List[RequestOutput] = []
        self.search_tasks = {}
        self.search_durations = {}
        self.embed_durations = {}
        self.ann_durations = {}
        self.thread_durations = {}
        self.start_engine_time = time.time()
        self.requests_token_time: Dict[str, List[List[float]]] = {}
        self.successive_requests: List[List[str]] = []
        self.requests_arr_time: Dict[str, float] = {}
        self.request_questions: Dict[str, str] = {}
        self.prefix_record: Dict[str, List[int]] = {}
        self.prefix_monitor_pointer: int = 0
        self.granularity_ls: List[int] = []
        self.if_terminate: Dict[str, int] = {}
        # prompt_list = prompt_list[:self.exp_config.max_prompt_num]
        self.executor = ThreadPoolExecutor(max_workers=5)

        self.exp_config._log()
        self.llm_engine.scheduler.priority_schedule = self.exp_config.priority_schedule

        while self.llm_engine.has_unfinished_requests() or self.search_tasks or arrival_requests:
            curr_time = time.time()
            if curr_time - self.start_engine_time > self.exp_config.test_duration:
                break
            
            while arrival_requests and arrival_requests[0][2] <= curr_time:
                self._add_request(prompt=arrival_requests[0][0], question=arrival_requests[0][1], arr_time=arrival_requests[0][2])
                arrival_requests.pop(0)
            
            if self.exp_config.non_stall_search and self.search_tasks and len(self.llm_engine.scheduler.waiting) > 0:
                terminated_requests = await self._check_stall_search()
            else:
                terminated_requests = []
            assert set(terminated_requests).issubset(self.search_tasks.keys()), f"Stalled requests not in search_tasks: {terminated_requests} not in {self.search_tasks.keys()}"

            # 总是先获取当前完成的任务
            done_tasks = [req_id for req_id, task in self.search_tasks.items() if task.done()]

            # 如果有指定终止请求，等待它们完成
            if terminated_requests:
                # 找出那些还没完成的任务
                pending_tasks = [
                    self.search_tasks[req_id]
                    for req_id in terminated_requests
                    if req_id in self.search_tasks and req_id not in done_tasks
                ]
                if pending_tasks:
                    await asyncio.wait(pending_tasks, return_when=asyncio.ALL_COMPLETED)

                # 更新 done_tasks，再次确认全部终止任务都完成
                done_tasks = [req_id for req_id, task in self.search_tasks.items() if task.done()]

            for task_id in done_tasks:
                req_id_old, new_ctx, max_possi_prefix_block_len, finish_time = self.search_tasks[task_id].result()
                root = Exp_Output.find_root(self.successive_requests, req_id_old)
                req_id_new = self._add_request(prompt=new_ctx, max_possi_prefix_block_len=max_possi_prefix_block_len, arr_time=finish_time, submit_time=self.requests_arr_time[root[0]])
                root.append(req_id_new)
                del self.search_tasks[task_id]

            if self.llm_engine.has_unfinished_requests():
                self._priority_schedule()
                loop = asyncio.get_event_loop()
                step_outputs, scheduler_outputs = await loop.run_in_executor(self.executor, self.llm_engine.step)

                for output in step_outputs:
                    ctx = output.prompt + output.outputs[0].text
                    query = self._check_for_search(ctx)
                    if query:
                        task = asyncio.create_task(self._search_query(output.request_id, ctx, query))
                        self.search_tasks[output.request_id] = task
                        self.llm_engine.abort_request(output.request_id)
                    elif QAData.extract_answer(output.outputs[0].text):
                        self.outputs.append(output)
                        self.llm_engine.abort_request(output.request_id)
                    elif output.finished:
                        self.outputs.append(output)
                self._record_for_one_step(scheduler_outputs)

            await asyncio.sleep(0.01)

        for req_id in self.requests_arr_time.keys():
            self.llm_engine.abort_request(req_id)
        for task in self.search_tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.search_tasks.values(), return_exceptions=True)
        return Exp_Output.from_llm(self)

    def _check_for_search(self, text: str) -> Optional[str]:
        text = text.strip()
        if text.endswith(SEARCH_LABLE_B):
            search_start = text.rfind(SEARCH_LABLE_A)
            if search_start != -1:
                return text[search_start + 8:text.rfind("</search>")].strip()
        return None

    async def _search_query(self, request_id: str, ctx: str, query: str, session=None) -> Tuple[str, str]:
        url = "http://localhost:8500/embed_and_retrieve"

        t1 = time.time()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1000)) as session:
            payload = {"request_id": request_id, "prompts": [query], "top_k": self.exp_config.topk, "anns_search_range":self.exp_config.anns_search_range,}
            async with session.post(url=url, json=payload, headers={"Content-Type": "application/json"}) as response:
                assert response.status == 200, f"Search Server Error for req {request_id}: {response.status}"
                res = await response.json()
                neighbor_texts = res['neighbor_texts']

        finish_time = time.time()
        self.search_durations[request_id] = finish_time - t1
        self.if_terminate[request_id] = 1 if res['is_stop_by_stall'] else 0
        self.embed_durations[request_id] = res['t_embed']
        self.ann_durations[request_id] = res['t_ann']
        self.thread_durations[request_id] = res['t_thread']

        # 3. 构造返回信息
        info = '\n'.join([f"(Doc {i+1})" + t.replace('\n',' ') for i,t in enumerate(neighbor_texts[0])])  # neighbor_texts 是一个列表的列表
        max_possi_prefix_len = len(self.llm_engine.tokenizer.tokenizer.tokenize(ctx))
        max_possi_prefix_block_len = max_possi_prefix_len // self.llm_engine.cache_config.block_size
        return request_id, f"{ctx}\n {INFORMATION_LABLE_A} {info} {INFORMATION_LABLE_B} \n", max_possi_prefix_block_len, finish_time

    def prefix_monitor(self,):
        granu = 50
        valid_prefix_records = np.array([v for v in self.prefix_record.values() if len(v) > 1])
        if len(valid_prefix_records) >= granu * (self.prefix_monitor_pointer + 1):
            self.cur_prefix_hit_rate = sum(valid_prefix_records[-granu:,1]) / sum(valid_prefix_records[-granu:,0])
            self.prefix_monitor_pointer += 1
            logging.info(f"{self.prefix_monitor_pointer} {self.cur_prefix_hit_rate}")

    def reset_search(self):
        url = "http://localhost:8500/reset"
        response = requests.get(url)
        return response.status_code == 200

    async def _check_stall_search(self, session=None):
        url = "http://localhost:8500/check_stall"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params={"step_low_ema": 0.90}) as response:
                assert response.status == 200, f"Check stall search error: {response.status}"
                stopeed_requests = await response.json()
        return [str(r) for r in stopeed_requests['stopeed_requests']]

    def del_engine(self):
        del self.llm_engine.model_executor.driver_worker
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    data = QAData(DATA_LIST[:1],1000000)

    config = Exp_Config(model=MODEL_LIST[0], max_model_len=20480, topk=TOPK, max_prompt_num=MAX_PROMPT_NUM, enable_prefix_cache=True, request_rate=REQUEST_RATE, priority_schedule=6)
    if IS_INSTRUCT: llm0 = LLM(exp_config=Exp_Config(model=MODEL_LIST[0]))
    prompt_list = []
    for question, answer in data.get_pairs(): 
        if IS_INSTRUCT:
            messages = [{"role": "system", "content": PROMPT_TEMPLATE}, {"role": "user", "content": question}]
            formatted_prompt = llm0.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = PROMPT_TEMPLATE.format(question)
        prompt_list.append((formatted_prompt, question))
    np.random.shuffle(prompt_list) # 7405

    req_rate = config.request_rate
    prompt_list_sub = prompt_list[:config.max_prompt_num]
    random_delays = np.random.exponential(1.0 / req_rate, len(prompt_list_sub))
    arrival_times = np.cumsum(random_delays)

    now = time.time()
    arrival_requests = [(prompt[0], prompt[1], now+arrival_times[i]) for i, prompt in enumerate(prompt_list_sub)]
    llm = LLM(exp_config=config)
    llm.reset_search()
    exp_output = asyncio.run(llm.run_engine_with_search(arrival_requests))
    total_answer, acc = data.accuracy(exp_output.q2o)
    logging.info(
        f"\nTest Duration: {time.time()-llm.start_engine_time}; "
        f"Total Answer: {total_answer}; "
        f"Accuracy: {acc}; "
    )
    exp_output.log_metrics()
    llm.del_engine()