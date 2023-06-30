import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import pandas as pd

import unittest
from unittest.mock import patch
from src.threshold import mean_std
from src.wrappers import spearmanr, pearsonr, l1, l2, euclidean_similarity
from src.cleaning import CleaningParam, Cleaning

class TestSomething(unittest.TestCase):
        
    def test_threshold(self):
        pred = [0.1, 0.3, 0.6, 0.8, 0.35, 0.74]
        t = 0.5
        # threshold = 0.6071795035488845
        # return with [0, 0, 0, 1, 0, 1]
        list_pass_threshold = mean_std(predictions=pred, t=t)
        list_pred_pass_threshold = []
        for i, th in enumerate(list_pass_threshold):
            if th:
                list_pred_pass_threshold.append(pred[i])
        self.assertEqual(list_pred_pass_threshold, [0.8, 0.74])
    
    def test_spearmanr_pearsonr_in_wrappers(self):
        x = np.array([7.1, 7.1, 7.2, 8.3, 9.4, 10.5, 11.4])
        y = np.array([2.8, 2.9, 2.8, 2.6, 3.5, 4.6, 5.0])
        self.assertEqual(spearmanr(x, y), 0.7000000000000001)
        self.assertEqual(pearsonr(x, y), 0.9347467974524515)

    def test_l1_l2_in_wrappers(self):
        v = torch.tensor([1.0, 2.0, 3.0])        
        self.assertTrue(torch.equal(torch.round(l1(v), decimals=4), torch.tensor([0.1667, 0.3333, 0.5000])))
        self.assertTrue(torch.equal(torch.round(l2(v), decimals=4), torch.tensor([0.2673, 0.5345, 0.8018])))

    def test_euclidean_similarity_in_wrappers(self):
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        c = np.array([1, 1, 0])
        self.assertEqual(euclidean_similarity(a, b), -1.4142135623730951)
        self.assertEqual(euclidean_similarity(c, b), -1.0)

    def test_cleaning(self):
        mock_agreements=pd.DataFrame({'lemma': ['apple', 'banana', 'cherry', 'grape'],
                                       'a': [0.8, 0.2, 0.2, 0.3],
                                       'b': [0.9, 0.3, 0.6, 0.9]})
        CP_above = CleaningParam(threshold=0.7, keep='above')
        CP_below = CleaningParam(threshold=0.5, keep='below')
        C_all_apple = Cleaning(stats={"a": CP_above, "b": CP_above}, match='all')
        C_all_grape = Cleaning(stats={"a": CP_below, "b": CP_above}, match='all')
        C_any = Cleaning(stats={"a": CP_below, "b": CP_below}, match='any') 

        self.assertEqual(list(C_all_apple(agreements=mock_agreements).lemma), ['apple'])
        self.assertEqual(list(C_all_grape(agreements=mock_agreements).lemma), ['grape'])
        self.assertEqual(list(C_any(agreements=mock_agreements).lemma),  ['banana', 'cherry', 'grape'])

if __name__ == '__main__':
    unittest.main()

