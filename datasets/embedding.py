import datasets
from sentence_transformers import SentenceTransformer
import numpy as np
from multiprocessing import Process, Queue
import torch
import sys

def encode_on_gpu(sentences, start_idx, end_idx, gpu_id, memmap, queue, model_path):
    model = SentenceTransformer(model_path, device=f'cuda:{gpu_id}')
    print(f'GPU {gpu_id} Model loaded OK')

    part_embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=384 
    )

    memmap[start_idx:end_idx] = part_embeddings
    queue.put((gpu_id, part_embeddings.shape))
    print(f'GPU {gpu_id} Part Ok: {part_embeddings.shape}')

def main(model_path, data_file_path, embedding_save_path):
    # num_gpus = torch.cuda.device_count()
    # if num_gpus < 2:
    #     raise ValueError("Need at least 2 GPUs!")

    # print(f'Using {num_gpus} GPUs!')

    corpus = datasets.load_dataset(
        'json',
        data_files=data_file_path,
        split="train",
        num_proc=32,
    )
    corpus = corpus.select_columns(['contents'])
    sentences = corpus['contents']
    print(f'Data loaded OK {len(corpus)}')
    del corpus

    num_parts = 2
    part_size = len(sentences) // num_parts

    embedding_dim = 384
    embeddings_memmap = np.memmap(embedding_save_path, dtype='float32', mode='w+', shape=(len(sentences), embedding_dim))

    queue = Queue()
    processes = []

    for part in range(num_parts):
        start_idx = part * part_size
        end_idx = (part + 1) * part_size if part < num_parts - 1 else len(sentences)
        part_sentences = sentences[start_idx:end_idx]
        print(f'Sentence Part {part} assigned to GPU {part} ({len(part_sentences)} sentences)...')

        p = Process(target=encode_on_gpu, args=(part_sentences, start_idx, end_idx, part, embeddings_memmap, queue, model_path))
        processes.append(p)
        p.start()

    for _ in range(num_parts):
        gpu_id, shape = queue.get()
        print(f'GPU {gpu_id} completed with shape {shape}')

    for p in processes:
        p.join()

    embeddings_memmap.flush()
    print("Embeddings saved to disk")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: embedding.py <SentenceTransformer_model_path> <data_file_path> <embedding_save_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    data_file_path = sys.argv[2]
    embedding_save_path = sys.argv[3]
    main(model_path, data_file_path, embedding_save_path)