<div align="center">    
  <img src="./logo.png" alt="SearchAgent-X Logo" width="40%"> <br>
</div>

***

<p align="center">
  <strong>SearchAgent-X</strong> is a highly efficient system for reasoning-search interleaved large language model (LLM) agents. <br>
  Compared to the popular LLM inference framework vLLM and HNSW-based retrieval methods, it achieves <strong>1.3–3.4×</strong> higher throughput with only <strong>0.2–0.6×</strong> the latency.
</p >

<div align="center">    
  <img src="./performance.png" alt="SearchAgent-X Performance" width="90%">
</div>

---

## 🚀 Quick Start

### Environment
- Retriever (and Encoder)
  ```bash
  conda create -n retriever_env python=3.12.9
  pip install -r retriever_requirements.txt
  ```
- Generator
   ```bash
   conda create -n SearchAgent-X python=3.9
   pip install -r generator_requirements.txt
   ```

### Datasets & Models
SearchAgent-X requires these datasets and models for running interleaved search and reasoning. Here we introduce our experimental settings. You can definitely change them to your own datasets/models. Remember where you store them for later configuration.
- Corpus:  [wiki-18-corpus](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/7023d43bec2094aa1a9470d3b25f0d702e12ca4a/retrieval-corpus)
- Embedding Model:  [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- ANN Index:  See [Build HNSW Index](#annindex) for details to build the HNSW index.
- LLM Reasoning Model:  [Search-R1-7B model](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo/commit/44ac5ffefbee4d7d32890066e6f3888ad7a273a1); [Search-R1-14B model](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo-v0.2)
- Request Dataset:  [Musique](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/musique)

### Run SearchAgent-X
- Modify the paths to your downloaded embedding model, HNSW index, and corpus in `config.py`
- Start Retriever Server  
   ```bash
   conda activate retriever_env
   python vllm/entrypoints/emb_ret_server.py
   ```
- Modify the paths to your downloaded datasets and models in `config.py`
- Run experiments
   ```bash
   conda activate SearchAgent-X
   python vllm/entrypoints/searchagent-x.py
   ```
   The experimental results will be stored by default in the directory `experiments/output/`.


## 👨‍💻 For Developers
### How To Encode And Index My Own Corpus?
The `dataset` directory contains scripts for processing your corpus: `embedding.py` for generating sentence embeddings and `build_hnsw.py` for constructing the HNSW index.

Follow these steps to prepare your corpus and build the search index:

1.  **Encode Corpus:**
    Use `embedding.py` to convert the corpus into embeddings using a specified Sentence Transformer model.

    ```bash
    python ./datasets/embedding.py <SentenceTransformer_model_path> <data_file_path> <embedding_save_path>
    ```
    * `<SentenceTransformer_model_path>`: Path to your specified Sentence Transformer model.
    * `<data_file_path>`: Path to your input data file (e.g., a `.jsonl` corpus).
    * `<embedding_save_path>`: Desired path to save the generated embeddings.

2.  **Build HNSW Index:** <a id="annindex"></a>
    Use `build_hnsw.py` to create an HNSW index for retrieval. You need to specify the `num_elements` and `data_dim` within the `build_hnsw.py` script based on your generated embeddings.

    ```bash
    python ./datasets/build_hnsw.py <embeddings_data_path> <hnsw_index_path>
    ```
    * `<embeddings_data_path>`: Path to the embeddings file generated in the previous step.
    * `<hnsw_index_path>`: Desired path to save the HNSW index file.
### How To Use Other Reasoning Models?
You can integrate different reasoning models by editing the `config.py`. Specifically, you'll need to:
1.  Set the `MODEL` path to your desired reasoning model.
2.  Configure the appropriate prompt template for that model within `config.py`.
### How To Deploy SearchAgent-X in Offline/Online Scenarios?
* **Offline Deployment:**
    Ideal for batch processing or scenarios where rate limiting isn't needed.
    Set `REQUEST_RATE = 'inf'` in `config.py`.

* **Online Deployment:**
    Designed for real-time applications where you need to manage request rate.
    Set `REQUEST_RATE` (requests per second) to a specific numerical value (e.g., `5`) in `config.py`.

Then, simply execute SearchAgent-X.

## 📋 What's Next?
1. Integrating SearchAgent-X into post-training frameworks like [Search-R1](https://github.com/petergriffinjin/search-r1), [ReSearch](https://github.com/Agent-RL/ReCall?tab=readme-ov-file), and [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher), measuring end-to-end training benefits.
2. Supporting more commonly used retrieval methods, such as IVF_PQ and SCANN.
3. ... (Expecting Your Feedback 😄!