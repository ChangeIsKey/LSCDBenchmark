import unittest
from unittest.mock import patch
import pandas as pd

from src.use import Use

class TestUse(unittest.TestCase):
    mock_data_arm = {'lemma': 'arm', 
                       'pos': 'N', 
                       'grouping': '1', 
                       'identifier': 'A', 
                       'context_preprocessed': 'and taking a knife from her pocket, she opened a vein in her little arm, and dipping a feather in the blood, wrote something on a piece of white cloth, which was spread before her.', 
                       'target_index_begin': 68,
                       'target_index_end': 71}
    mock_series_arm = pd.Series(data=mock_data_arm, index=['lemma', 'pos', 'grouping', 'identifier', 'context_preprocessed', 'target_index_begin', 'target_index_end']) 
    U = Use(identifier=str(),
                grouping=str(),
                context=str(),
                target=str(),
                indices=(int(), int()),
                pos=str())
    def test_from_series(self):
        use_arm = self.U.from_series(self.mock_series_arm)
        self.assertEqual(use_arm.target, 'arm')
        self.assertEqual(use_arm.pos, 'N')
        self.assertEqual(use_arm.grouping, '1')
        self.assertEqual(use_arm.identifier, 'A')
        self.assertEqual(use_arm.context, 'and taking a knife from her pocket, she opened a vein in her little arm, and dipping a feather in the blood, wrote something on a piece of white cloth, which was spread before her.')
        self.assertEqual(use_arm.indices, (68, 71))

    @patch('src.use.hash')
    def test___hash__(self, mock_hash):
        use_arm = self.U.from_series(self.mock_series_arm)
        use_arm.__hash__()
        self.assertTrue(mock_hash.called)

    def test___hash___return_int(self):
        use_arm = self.U.from_series(self.mock_series_arm)
        hash = use_arm.__hash__()
        self.assertIsInstance(hash, int)
    
    def test___lt__(self):
        mock_data_100 = {'lemma': '', 
                         'pos': '', 
                         'grouping': '', 
                         'identifier': '100', 
                         'context_preprocessed': '', 
                         'target_index_begin': 0,
                         'target_index_end': 0}
        use_100 = self.U.from_series(pd.Series(data=mock_data_100, index=['lemma', 'pos', 'grouping', 'identifier', 'context_preprocessed', 'target_index_begin', 'target_index_end']))
        mock_data_500 = {'lemma': '', 
                         'pos': '', 
                         'grouping': '', 
                         'identifier': '500', 
                         'context_preprocessed': '', 
                         'target_index_begin': 0,
                         'target_index_end': 0}
        use_500 = self.U.from_series(pd.Series(data=mock_data_500, index=['lemma', 'pos', 'grouping', 'identifier', 'context_preprocessed', 'target_index_begin', 'target_index_end']))
        lt_return = use_100.__lt__(other=use_500)
        self.assertIsInstance(lt_return, bool)
        self.assertTrue(lt_return)
