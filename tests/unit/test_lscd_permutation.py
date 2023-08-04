import sys
sys.path.insert(0, ".")

import unittest
from unittest.mock import patch
from scipy.spatial.distance import cosine
import numpy as np
import os
import pathlib

from src.lscd.permutation import Permutation
from src.wic.contextual_embedder import ContextualEmbedder
from src.lemma import Lemma
from src.preprocessing import Lemmatize
from src.dataset import Dataset

class TestPermutation(unittest.TestCase):
    WIC = ContextualEmbedder(truncation_tokens_before_target=0.5,
                            similarity_metric=cosine, 
                            ckpt="bert-base-uncased",
                            layers=[1, 12],
                            embedding_cache=None,
                            layer_aggregation="average",
                            subword_aggregation="average",
                            encode_only=False
                            )
    P = Permutation(wic=WIC, n_perms=3, whiten=False, k=3)
    
    # test with testwug_en_111, check if dataset exist
    if os.path.isdir('wug/testwug_en_111'):
        print('testwug_en_111 is ready!')
    else:
        print('testwug_en_111 does not exist')
        D = Dataset(path='testwug_en_111', 
                    groupings=tuple(['1', '2']), 
                    type='dev',
                    split='dev',
                    exclude_annotators=[],
                    name='testwug_en_111',
                    test_on=None,
                    cleaning=None,
                    url='https://zenodo.org/record/7946753/files/testwug_en.zip')
        D._Dataset__download_zip()
        print('testwug_en_111 is downloaded!')

    mock_testwug_path_0 = pathlib.Path('wug/testwug_en_111/data/afternoon_nn')
    mock_testwug_path_1 = pathlib.Path('wug/testwug_en_111/data/plane_nn')

    Lem = Lemmatize(spelling_normalization=None) 
    Lemma_0 = Lemma(groupings=('1', '2'), 
              path=mock_testwug_path_0,
              preprocessing=Lem)
    Lemma_1 = Lemma(groupings=('1', '2'), 
              path=mock_testwug_path_1,
              preprocessing=Lem)
    
    def test_compute_kernel_bias_k_3(self):
        vecs = np.array([[0.11, 0.12, 0.13, 0.14], [0.21, 0.22, 0.23, 0.24]])
        w, m = self.P.compute_kernel_bias(vecs=vecs, k=3)
        self.assertEqual(w.shape, (4, 3))
        self.assertEqual(round(m[0][0], 2), -0.16)
        self.assertEqual(round(m[0][1], 2), -0.17)
        self.assertEqual(round(m[0][2], 2), -0.18)
        self.assertEqual(round(m[0][3], 2), -0.19)
    
    def test_compute_kernel_bias_k_None(self):
        vecs = np.array([[0.11, 0.12, 0.13, 0.14], [0.21, 0.22, 0.23, 0.24]])
        w, m = self.P.compute_kernel_bias(vecs=vecs, k=None)
        self.assertEqual(w.shape, (4, 4))
        self.assertEqual(round(m[0][0], 2), -0.16)
        self.assertEqual(round(m[0][1], 2), -0.17)
        self.assertEqual(round(m[0][2], 2), -0.18)

    def test_transform_and_normalize(self):
        vecs = np.array([[0.11, 0.12, 0.13, 0.14], [0.21, 0.22, 0.23, 0.24]])
        result = self.P.transform_and_normalize(vecs=vecs, kernel=None, bias=None)
        self.assertEqual(round(result[0][0], 2), 0.44)
        self.assertEqual(round(result[0][1], 2), 0.48)
        self.assertEqual(round(result[1][2], 2), 0.51)
        self.assertEqual(round(result[1][3], 2), 0.53)
        
    def test_euclidean_dist(self):
        m0 = np.array([[0.11, 0.12, 0.13, 0.14], [0.21, 0.22, 0.23, 0.24]])
        m1 = np.array([[0.31, 0.32, 0.33, 0.34], [0.41, 0.42, 0.43, 0.44]])
        result = self.P.euclidean_dist(m0=m0, m1=m1)
        self.assertEqual(round(result[0][0], 2), 0.16)
        self.assertEqual(round(result[0][1], 2), 0.36)
        self.assertEqual(round(result[1][0], 2), 0.04)
        self.assertEqual(round(result[1][1], 2), 0.16)

    def test_get_n_rows(self):
        m0_less_m1 = self.P.get_n_rows(len_m0=3, len_m1=7)
        self.assertIsInstance(m0_less_m1, int)
        m0_grater_m1 = self.P.get_n_rows(len_m0=10, len_m1=3)
        self.assertIsInstance(m0_grater_m1, int)

    def test_permute_indices(self):
        two_lsts = self.P.permute_indices(len_m0=3, len_m1=7)
        self.assertEqual(3, len(two_lsts[0]))
        self.assertEqual(7, len(two_lsts[1]))

    def test_predict(self):
        result = self.P.predict(lemma=self.Lemma_0)
        print(result)
        # TODO: no error, but get 0.0
        # self.assertEqual(, round(result, 4))
    
    def test_predict_all(self):
        self.P.predict_all(lemmas=[self.Lemma_0, self.Lemma_1])
        # TODO: same error for all 'predict_all'
        # TypeError: Object of type 'function' is not JSON serializable

if __name__ == '__main__':
    unittest.main()

