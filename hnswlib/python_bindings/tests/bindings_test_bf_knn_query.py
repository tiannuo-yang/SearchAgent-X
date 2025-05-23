import unittest

import numpy as np
import time
import hnswlib


class RandomSelfTestCase(unittest.TestCase):
    def testIndexBf(self):

        dim = 160
        num_elements = 200000
        num_queries = 100
        k = 20

        # Generating sample data
        data = np.float32(np.random.random((num_elements, dim)))

        # Declaring index
        index = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
        index.init_index(max_elements=num_elements)

        index.set_num_threads(64)  # threads for build

        print(f"Adding all elements {num_elements}")
        index.add_items(data)

        print("Checking results")
        num_threads = 12
        index.set_num_threads(num_threads)  # threads for query

        t1 = time.time()
        for i in range(num_queries):
            print(i)
            query = np.float32(np.random.random((1, dim)))
            labels_bf, dists_bf = index.bf_knn_query(query, k=k)
            sq_dists = (data - query)**2
            dists = np.sum(sq_dists, axis=1)
            labels_gt = np.argsort(dists)[:k]
            dists_gt = dists[labels_gt]
            # we can compare labels but because of numeric errors in distance calculation in C++ and numpy
            # sometimes we get different order of labels, therefore we compare distances
            max_diff_with_gt = np.max(np.abs(dists_gt - dists_bf))
            if max_diff_with_gt > 1e-5:
                breakpoint()
            self.assertTrue(max_diff_with_gt < 1e-5)
        t2 = time.time()
        print(f"Time taken for {num_queries} queries: {t2 - t1}")

if __name__ == "__main__":
    unittest.main()