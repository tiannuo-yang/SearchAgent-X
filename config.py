# 1. retriever config
EMBEDDER = '/root/data/Dataset/all-MiniLM-L6-v2'
HNSW_INDEX = '/root/data/Dataset/hnsw_index_efc500_m32.bin'
CORPUS = '/root/data/Dataset/wiki-18-corpus/wiki-18.jsonl'

# 2. generator config
# Reasoning Model and Data Paths
MODEL = '/root/data/Dataset/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo'
DATA = '/root/data/Dataset/FlashRAG_datasets/musique/dev.jsonl'
# Prompt Template Configuration
# qwen2.5-7b prompt template for example
PROMPT_TEMPLATE = (
    "Answer the given Question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
    "Question: {}"
)
SEARCH_LABLE_A = '<search>'
SEARCH_LABLE_B = '</search>'
INFORMATION_LABLE_A = '<information>'
INFORMATION_LABLE_B = '</information>'
IS_INSTRUCT = False

# 3. Experiment Setups
OUTPUT_FILE = '/root/SearchAgent-X/experiments/output/test.log'
REQUEST_RATE = 'inf' # REQUEST_RATE > 0 for online, REQUEST_RATE = 'inf' for offline
TEST_DURATION = 600 # seconds
MAX_PROMPT_NUM = 5000
TOPK = 5
TENSOR_PARALLEL_SIZE = 1
PRIORITY_SCHEDULE = 6
RETRIEVER = 1 # 0 for ENN, 1 for ANN 