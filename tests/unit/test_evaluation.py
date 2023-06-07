import sys
sys.path.insert(0, ".")

from src.evaluation import Evaluation

import io
import os
from unittest.mock import MagicMock, patch, Mock, mock_open
import unittest

class TestDataset(unittest.TestCase):
    
    def test_combine_input(self):
        mock_labels = {'key_1': 1,
                       'key_2': 2,
                       'key_3': 3}
        
        mock_preds = {'key_1': 3,
                      'key_2': 5,
                      'key_3': 6}
        
        merged = Evaluation.combine_inputs(labels=mock_labels, predictions=mock_preds)
        self.assertEqual(list(merged['instance']), ['key_1', 'key_2', 'key_3'])
        self.assertEqual(list(merged['prediction']), [3, 5, 6])
        self.assertEqual(list(merged['label']), [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
