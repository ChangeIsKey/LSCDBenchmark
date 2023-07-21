import sys
sys.path.insert(0, ".")

from src.evaluation import Evaluation
from src.wrappers import spearmanr
from src.plots import Plotter

import os
import unittest

class TestEvaluation(unittest.TestCase):
    P = Plotter(max_alpha=0.5, 
                default_alpha=0.3, 
                min_boots_in_one_tail=2, 
                metric=spearmanr)
    E = Evaluation(task="binary_wic", metric=spearmanr, plotter=P)
    mock_labels = {'key_1': 1,
                   'key_2': 2,
                   'key_3': 3}
        
    mock_preds = {'key_1': 3,
                  'key_2': 5,
                  'key_3': 6}

    def test_combine_input(self):
        merged = self.E.combine_inputs(predictions=self.mock_preds, labels=self.mock_labels)
        self.assertEqual(list(merged['instance']), ['key_1', 'key_2', 'key_3'])
        self.assertEqual(list(merged['prediction']), [3, 5, 6])
        self.assertEqual(list(merged['label']), [1, 2, 3])

    def test_func_in___call__(self):
        print(spearmanr.__name__) # 'spearmanr'
        print(self.E(predictions=self.mock_preds, labels=self.mock_labels))
        # TODO: run python -m unittest tests.unit.test_evaluation.test_func_in___call__
        
        self.assertEqual(self.E(predictions=self.mock_preds, labels=self.mock_labels), 1.0)
        os.remove("result.json")
        os.remove("histogram.png")
        os.remove("histogram.svg")

if __name__ == '__main__':
    unittest.main()
