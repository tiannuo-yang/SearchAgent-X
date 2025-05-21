import hnswlib
import numpy as np
import time
import sys

def build_hnsw_index(embeddings_data_path, hnsw_index_path):
    """
    Builds an HNSW index from embeddings data and saves it to a specified path.

    Args:
        embeddings_data_path (str): Path to the memory-mapped embeddings data file.
        hnsw_index_path (str): Path to save the built HNSW index.
    """

    # Reading embeddings
    t0 = time.time()
    # You'll need to know num_elements and data_dim beforehand.
    num_elements, data_dim = 21015324, 384
    
    try:
        embeddings_memmap = np.memmap(embeddings_data_path, dtype='float32', mode='r', shape=(num_elements, data_dim))
    except FileNotFoundError:
        print(f"Error: Embeddings data file not found at '{embeddings_data_path}'")
        return
    except Exception as e:
        print(f"Error loading embeddings data: {e}")
        return

    print(f'Data loaded OK: Total {time.time()-t0:.2f}s')

    p = hnswlib.Index(space='l2', dim=data_dim)

    p.init_index(max_elements=num_elements, ef_construction=500, M=32)

    p.set_num_threads(64)

    p.add_items(embeddings_memmap)
    print(f'Data Inserted OK: Total {time.time()-t0:.2f}s')

    try:
        p.save_index(hnsw_index_path)
        print(f'HNSW index saved successfully to: {hnsw_index_path}')
    except Exception as e:
        print(f"Error saving HNSW index: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: build_hnsw.py <embeddings_data_path> <hnsw_index_path>")
        sys.exit(1)

    embeddings_data_path = sys.argv[1]
    hnsw_index_path = sys.argv[2]

    build_hnsw_index(embeddings_data_path, hnsw_index_path)