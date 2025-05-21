import unittest

import numpy as np
import time
import hnswlib
import sys
import threading


class RandomSelfTestCase(unittest.TestCase):
    def testIndexBf(self):

        dim = 100
        num_elements = 500000
        num_queries = 100
        k = 1

        # Generating sample data
        data = np.float32(np.random.random((num_elements, dim)))

        # Declaring index
        index = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)

        index.set_num_threads(64)  # threads for build

        print(f"Adding all elements {num_elements}")
        index.add_items(data)

        print("Starting queries")
        index.set_num_threads(1)
        index.set_ef(10000)
        querys = np.float32(np.random.random((num_queries, dim)))

        # 1. normal search
        labels1 = []
        start = time.time_ns()
        for i in range(num_queries):
            query = querys[i]
            labels, dists = index.knn_query(query, k=k)
            labels1.append(labels[0])
        end = time.time_ns()
        print(f"Time each taken for {num_queries} queries: {(end - start)/(num_queries*1000*1000)} ms")

        # 2. search with stall time
        labels2 = []
        is_stop = []
        def early_stop(step_num_stop):
            while True:
                index.early_stop(step_num_stop) # ☆☆☆
                time.sleep(1 / 1000)
        step_num_stop = 1000
        t = threading.Thread(target=early_stop, args=(step_num_stop,))
        t.deamon = True
        t.start()

        start = time.time_ns()
        for i in range(num_queries):
            query = querys[i]
            labels, dists, is_stop_by_stall = index.knn_query_with_stall(query, k=k) # ☆☆☆ one more return value
            labels2.append(labels[0])
            is_stop.append(is_stop_by_stall)
        end = time.time_ns()
        print(f"Early Stop num={step_num_stop} Time each taken for {num_queries} queries: {(end - start)/(num_queries*1000*1000)} ms")
        print(f"Successfully early stop {np.sum(is_stop)} queries")

        # 3. recall
        recall1 = 0
        recall2 = 0
        for i in range(num_queries):
            query = querys[i]
            sq_dists = (data - query)**2
            dists = np.sum(sq_dists, axis=1)
            labels_gt = np.argsort(dists)[:k]
            dists_gt = dists[labels_gt]
            recall1 += len(np.intersect1d(labels1[i], labels_gt))
            labels_gt2 = np.argsort(dists)[:k]
            dists_gt = dists[labels_gt2]
            recall2 += len(np.intersect1d(labels2[i], labels_gt2))
        print(f"recall1 = {recall1/(num_queries*k)}")
        print(f"recall2 = {recall2/(num_queries*k)}")

if __name__ == "__main__":
    unittest.main()