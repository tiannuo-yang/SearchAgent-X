import os
import sys
import logging
from typing import List, Dict
from vllm.outputs import RequestOutput
import numpy as np
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, config_dir)
from config import (
    OUTPUT_FILE,
    MODEL,
    DATA,
    PROMPT_TEMPLATE,
    SEARCH_LABLE_A,
    SEARCH_LABLE_B,
    INFORMATION_LABLE_A,
    INFORMATION_LABLE_B,
    IS_INSTRUCT,
    REQUEST_RATE,
    TEST_DURATION,
    MAX_PROMPT_NUM,
    TOPK,
    TENSOR_PARALLEL_SIZE,
    RETRIEVER
)

search_range = 0 if RETRIEVER == 0 else 10000
retriever = 'ENN' if RETRIEVER == 0 else 'ANN'
class Exp_Config:
    def __init__(
        self,
        request_rate: float = 2,
        query_time: float = 1.0,
        anns_search_range: int = search_range,
        topk: int = 3,
        test_duration: int = TEST_DURATION,
        max_tokens: int = 512,
        seed: int = 1206,
        enable_prefix_cache: bool = True,
        model: str = None,
        tensor_parallel_size: int = TENSOR_PARALLEL_SIZE,
        max_model_len: int = 16000,
        enforce_eager: bool = True,
        priority_schedule: int = -1, # 0 for search_count-only priority schedule, 1 for hierarchical priority schedule, -1 for vanilla schedule
        max_prompt_num: int = 5000,
        non_stall_search: bool = True,
        delay_factor: float = 0.5,
    ) -> None:
        """实验配置类，包含所有实验参数"""
        self.request_rate = request_rate
        self.query_time = query_time
        self.anns_search_range = anns_search_range
        self.topk = topk
        self.test_duration = test_duration
        self.max_tokens = max_tokens
        self.seed = seed
        self.enable_prefix_cache = enable_prefix_cache
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.priority_schedule = priority_schedule
        self.max_prompt_num = max_prompt_num
        self.non_stall_search = non_stall_search
        self.delay_factor = delay_factor

    def _log(self,):
        logging.info(
            f"Request Rate: {self.request_rate}; "
            f"Query Time: {self.query_time}; "
            f"Retriever: {retriever}; "
            f"Top K: {self.topk}; "
            f"Test Duration: {self.test_duration}; "
            f"Max Tokens: {self.max_tokens}; "
            f"Seed: {self.seed}; "
            f"Model: {self.model}; "
            f"Tensor Parallel Size: {self.tensor_parallel_size}; "
            f"Max Model Len: {self.max_model_len}; "
            f"Priority Schedule: {self.priority_schedule}; "
            f"Max Prompt Num: {self.max_prompt_num}; "
        )

class Exp_Output:
    def __init__(
            self,
            start_engine_time: float,
            outputs: List[RequestOutput],
            search_durations: Dict[str, float],
            embed_durations: Dict[str, float],
            ann_durations: Dict[str, float],
            thread_durations: Dict[str, float],
            requests_token_time: Dict[str, List[List[float]]],
            successive_requests: List[List[str]],
            requests_arr_time: Dict[str, float],
            request_questions: Dict[str, str],
            prefix_record: Dict[str, List[int]] = None,
            granularity_ls: List[int] = None,
            if_terminate: Dict[str, int] = {},
            ) -> None:
        self.start_engine_time = start_engine_time
        self.outputs = outputs
        self.search_durations = search_durations
        self.embed_durations = embed_durations
        self.ann_durations = ann_durations
        self.thread_durations = thread_durations
        self.requests_token_time = requests_token_time
        self.successive_requests = successive_requests
        self.requests_arr_time = requests_arr_time
        self.request_questions = request_questions
        self.prefix_record = prefix_record
        self.granularity_ls = granularity_ls if granularity_ls else []
        self.if_terminate = if_terminate

        self.calcu_metrics()

    @staticmethod
    def find_root(d: List[List[str]], id: str) -> str:
        """查找请求的根ID"""
        for req_ls in d:
            if id in req_ls:
                return req_ls

    def calcu_metrics(self) -> None:
        """计算并打印性能指标"""
        finished_reqs = [Exp_Output.find_root(self.successive_requests, o.request_id)[0] for o in self.outputs]
        finished_tokens = [self.requests_token_time[r] for r in finished_reqs]

        self.tret = [ts for tss in self.requests_token_time.values() for ts in tss if isinstance(ts, float)]
        self.ttft_first = [tss[0][0] for tss in self.requests_token_time.values()]
        self.ttft_resume = [ts[0] for tss in self.requests_token_time.values() for i,ts in enumerate(tss) if isinstance(ts, list) and i > 0]
        self.ttft_all = self.ttft_first + self.ttft_resume
        self.tpot = [t for tss in self.requests_token_time.values() for ts in tss if isinstance(ts, list) for t in ts[1:]]
        self.token_num = len(self.ttft_all) + len(self.tpot)
        self.te2e = [sum(sum(ts) if isinstance(ts, list) else ts for ts in tss) for tss in finished_tokens]
        self.prefix_complete = np.array([v for v in self.prefix_record.values() if len(v) == 2])
        self.finished_req_ret_counts = [len([ts for ts in tss if isinstance(ts, float)]) for tss in finished_tokens]
        self.finished_req_text_lens = [len(o.prompt + o.outputs[0].text) for o in self.outputs]

        self.sent_num_req = len(self.request_questions)
        self.total_num_ret = len(self.tret)
        self.finished_num_req = len(self.outputs)
        self.avg_t_search = sum(self.search_durations.values()) / len(self.search_durations) if self.search_durations else 0
        self.avg_t_embed = sum(self.embed_durations.values()) / len(self.embed_durations) if self.embed_durations else 0
        self.avg_t_ann = sum(self.ann_durations.values()) / len(self.ann_durations) if self.ann_durations else 0
        self.avg_t_thread = sum(self.thread_durations.values()) / len(self.thread_durations) if self.thread_durations else 0
        self.avg_t_ret = sum(self.tret) / len(self.tret) if self.tret else 0
        self.avg_ttft_first = sum(self.ttft_first) / len(self.ttft_first) if self.ttft_first else 0
        self.avg_ttft_resume = sum(self.ttft_resume) / len(self.ttft_resume) if self.ttft_resume else 0
        self.avg_ttft_all = sum(self.ttft_all) / len(self.ttft_all) if self.ttft_all else 0
        self.avg_tpot = sum(self.tpot) / len(self.tpot) if self.tpot else 0
        self.avg_te2e = sum(self.te2e) / len(self.te2e) if self.te2e else 0
        self.p99_te2e = np.percentile(self.te2e, 99) if self.te2e else 0
        self.p95_te2e = np.percentile(self.te2e, 95) if self.te2e else 0
        self.avg_ret_counts = sum(self.finished_req_ret_counts) / len(self.finished_req_ret_counts) if self.finished_req_ret_counts else 0
        self.avg_output_lens = sum(self.finished_req_text_lens) / len(self.finished_req_text_lens) if self.finished_req_text_lens else 0
        self.throughput = len(self.outputs) / (time.time() - self.start_engine_time)
        self.throughput_token = self.token_num / (time.time() - self.start_engine_time)
        self.prefix_hit_rate = sum(self.prefix_complete[:,1]) / sum(self.prefix_complete[:,0]) if len(self.prefix_complete) else 0
        self.terminate_ratio = sum(list(self.if_terminate.values())) / len(list(self.if_terminate.values()))

        self.q2o = [(self.request_questions[Exp_Output.find_root(self.successive_requests, o.request_id)[0]], o.prompt + o.outputs[0].text)
                for o in self.outputs]

    def log_metrics(self) -> None:
        logging.info(
            f"\nSent #Reqs: {self.sent_num_req}; Total #Ret: {self.total_num_ret}; Finished #Reqs: {self.finished_num_req}; \n"
            f"Avg TSearch: {self.avg_t_search} s; \n"
            f"Avg TEmbed: {self.avg_t_embed} s; \n"
            f"Avg TAnn: {self.avg_t_ann} s; \n"
            f"Avg TThread: {self.avg_t_thread} s; \n"
            f"Avg TRet: {self.avg_t_ret} s; \n"
            f"Avg TTFT (First): {self.avg_ttft_first} s; \n"
            f"Avg TTFT (Resume): {self.avg_ttft_resume} s; \n"
            f"Avg TTFT (All): {self.avg_ttft_all}; \n"
            f"Avg TPOT: {self.avg_tpot} s; \n"
            f"Avg TE2E: {self.avg_te2e} s; \n"
            f"P99 TE2E: {self.p99_te2e} s; \n"
            f"P95 TE2E: {self.p95_te2e} s; \n"
            f"Avg Ret Counts: {self.avg_ret_counts}; \n"
            f"Avg Output Length: {self.avg_output_lens}; \n"
            f"Throughput: {self.throughput} #seq/s \n"
            f"Throughput Token: {self.throughput_token} #token/s \n"
            f"Avg Prefix Hit Rate: {self.prefix_hit_rate}; \n"
            f"Terminate Ratio: {self.terminate_ratio}; \n"
            )

    def to_dict(self, q2o: bool = False) -> Dict:
        """将类的关键属性转换为字典"""
        return {
        "start_engine_time": self.start_engine_time,
        "search_durations": self.search_durations,
        "embed_durations": self.embed_durations,
        "ann_durations": self.ann_durations,
        "thread_durations": self.thread_durations,
        "requests_token_time": self.requests_token_time,
        "successive_requests": self.successive_requests,
        "requests_arr_time": self.requests_arr_time,
        "request_questions": self.request_questions,
        "prefix_record": self.prefix_record,
        "granularity_ls": self.granularity_ls,
        "tret": self.tret,
        "ttft_first": self.ttft_first,
        "ttft_resume": self.ttft_resume,
        "ttft_all": self.ttft_all,
        "tpot": self.tpot,
        "te2e": self.te2e,
        "sent_num_req": self.sent_num_req,
        "total_num_ret": self.total_num_ret,
        "finished_num_req": self.finished_num_req,
        "avg_t_search": self.avg_t_search,
        "avg_t_ret": self.avg_t_ret,
        "avg_t_embed": self.avg_t_embed,
        "avg_t_ann": self.avg_t_ann,
        "avg_t_thread": self.avg_t_thread,
        "avg_ttft_first": self.avg_ttft_first,
        "avg_ttft_resume": self.avg_ttft_resume,
        "avg_ttft_all": self.avg_ttft_all,
        "avg_tpot": self.avg_tpot,
        "avg_te2e": self.avg_te2e,
        "p99_te2e": self.p99_te2e,
        "p95_te2e": self.p95_te2e,
        "avg_ret_counts": self.avg_ret_counts,
        "avg_output_lens": self.avg_output_lens,
        "throughput": self.throughput,
        "throughput_token": self.throughput_token,
        "q2o": self.q2o if q2o else None,
        "prefix_hit_rate": self.prefix_hit_rate,
        "terminate_ratio": self.terminate_ratio
        }

    @classmethod
    def from_llm(cls, llm):
        """从LLM实例创建Exp_Output实例"""
        return cls(
            start_engine_time = llm.start_engine_time,
            outputs = llm.outputs,
            search_durations = llm.search_durations,
            embed_durations = llm.embed_durations,
            ann_durations = llm.ann_durations,
            thread_durations = llm.thread_durations,
            requests_token_time = llm.requests_token_time,
            successive_requests = llm.successive_requests,
            requests_arr_time = llm.requests_arr_time,
            request_questions = llm.request_questions,
            prefix_record = llm.prefix_record,
            granularity_ls = llm.granularity_ls,
            if_terminate = llm.if_terminate
        )
