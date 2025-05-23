import datasets
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import sys

def main(model_path, data_file_path, embedding_save_path):
    # Ensure a GPU is available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model = SentenceTransformer(model_path, device=device)
    print(f'Model loaded OK on {device}')

    corpus = datasets.load_dataset(
        'json',
        data_files=data_file_path,
        split="train",
        num_proc=32,
    )
    corpus = corpus.select_columns(['contents'])
    sentences = corpus['contents']
    print(f'Data loaded OK: {len(sentences)} sentences')
    del corpus

    embedding_dim = model.get_sentence_embedding_dimension() # Dynamically get embedding dimension
    if embedding_dim is None: # Fallback if method doesn't exist or returns None
        # Default to 384 or raise an error if model's dimension is unknown and critical
        print("Warning: Could not determine embedding dimension from model. Defaulting to 384.")
        embedding_dim = 384 

    embeddings_memmap = np.memmap(embedding_save_path, dtype='float32', mode='w+', shape=(len(sentences), embedding_dim))

    print(f'Starting embedding generation for {len(sentences)} sentences...')
    all_embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=384
    )

    embeddings_memmap[:] = all_embeddings
    embeddings_memmap.flush()
    print("Embeddings saved to disk")
    print(f"Total embeddings shape: {embeddings_memmap.shape}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: embedding.py <SentenceTransformer_model_path> <data_file_path> <embedding_save_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    data_file_path = sys.argv[2]
    embedding_save_path = sys.argv[3]
    main(model_path, data_file_path, embedding_save_path)