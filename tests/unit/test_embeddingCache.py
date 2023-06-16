import sys
sys.path.insert(0, ".")

import unittest
from unittest.mock import patch

from src.use import Use
from src.wic.contextual_embedder import EmbeddingCache
import pandas as pd
import torch
from pathlib import Path


class TestEmbeddingCache(unittest.TestCase):
    mock_data_arm_1 = {'lemma': 'arm', 
                       'pos': 'N', 
                       'grouping': '1', 
                       'identifier': 'A', 
                       'context_preprocessed': 'and taking a knife from her pocket, she opened a vein in her little arm, and dipping a feather in the blood, wrote something on a piece of white cloth, which was spread before her.', 
                       'target_index_begin': 68,
                       'target_index_end': 71}
    mock_series_arm_1 = pd.Series(data=mock_data_arm_1, index=['lemma', 'pos', 'grouping', 'identifier', 'context_preprocessed', 'target_index_begin', 'target_index_end']) 
    
    mock_data_arm_2 = {'lemma': 'arm', 
                       'pos': 'N', 
                       'grouping': '2', 
                       'identifier': 'D', 
                       'context_preprocessed': 'It stood behind a high brick wall, its back windows overlooking an arm of the sea which, at low tide, was a black and stinking mud-flat', 
                       'target_index_begin': 67, 
                       'target_index_end': 70}
    mock_series_arm_2 = pd.Series(data=mock_data_arm_2, index=['lemma', 'pos', 'grouping', 'identifier', 'context_preprocessed', 'target_index_begin', 'target_index_end']) 
    
    mock_data_target = {'lemma': 'target', 
                        'pos': 'N', 
                        'grouping': '2', 
                        'identifier': 'target-F', 
                        'context_preprocessed': 'something something something target something something.', 
                        'target_index_begin': 30, 
                        'target_index_end': 36}
    mock_series_target = pd.Series(data=mock_data_target, index=['lemma', 'pos', 'grouping', 'identifier', 'context_preprocessed', 'target_index_begin', 'target_index_end'])

    mock_data_empty = {'lemma': '', 
                        'pos': '', 
                        'grouping': '', 
                        'identifier': '', 
                        'context_preprocessed': '', 
                        'target_index_begin': 0, 
                        'target_index_end': 0}
    mock_series_empty = pd.Series(data=mock_data_empty, index=['lemma', 'pos', 'grouping', 'identifier', 'context_preprocessed', 'target_index_begin', 'target_index_end']) 
    
    U = Use(identifier=str(),
                grouping=str(),
                context=str(),
                target=str(),
                indices=(int(), int()),
                pos=str())
    
    mock_metadata = {'dataset.name': 'testwug_en_111',
                        'dataset.preprocessing': 'raw',
                        'contextual_embedder.pre_target_tokens': 0.0, 
                        'contextual_embedder.ckpt': 'mock_model_name',
                        'id': 'mock_id', 
                        'target': 'arm'}

    def test_new_cache_added_with_new_target_and_embedding(self):
        use_arm = self.U.from_series(self.mock_series_arm_1)
        use_target = self.U.from_series(self.mock_series_target)

        E = EmbeddingCache(metadata=dict())
        
        self.assertEqual(E._targets_with_new_uses, set())
        self.assertEqual(E._cache, {})

        E.add_use(use=use_arm, embedding=torch.Tensor())
        self.assertEqual(E._targets_with_new_uses, {'arm'})
        # E._cache = {'arm': {'A': tensor([])}}
        self.assertEqual(list(E._cache['arm'].keys()), ['A'])
        self.assertIsInstance(list(E._cache['arm'].values())[0], torch.Tensor)

        E.add_use(use=use_target, embedding=torch.tensor([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(E._targets_with_new_uses, {'arm', 'target'})
        # E._cache = {'arm': {'A': tensor([])}, 'target': {'target-F': tensor([])}}
        self.assertEqual(list(E._cache.keys()), ['arm', 'target'])
        self.assertTrue(torch.equal(list(E._cache['target'].values())[0], torch.tensor([[1, 2, 3], [4, 5, 6]])))
        self.assertIsInstance(list(E._cache['target'].values())[0], torch.Tensor)
    
    def test_new_cache_added_with_same_target_differnt_identifier(self):
        use_arm_1 = self.U.from_series(self.mock_series_arm_1)
        use_arm_2 = self.U.from_series(self.mock_series_arm_2)

        E = EmbeddingCache(metadata=dict())
        
        self.assertEqual(E._targets_with_new_uses, set())
        self.assertEqual(E._cache, {})

        E.add_use(use=use_arm_1, embedding=torch.Tensor())
        self.assertEqual(E._targets_with_new_uses, {'arm'})
        # E._cache = {'arm': {'A': tensor([])}}
        self.assertEqual(list(E._cache['arm'].keys()), ['A'])

        E.add_use(use=use_arm_2, embedding=torch.Tensor())
        self.assertEqual(E._targets_with_new_uses, {'arm'})
        # E._cache = {'arm': {'A': tensor([]), 'D': tensor([])}}
        self.assertEqual(list(E._cache['arm'].keys()), ['A', 'D'])
    
    def test_no_cache_added_with_same_identifier(self):
        use_arm_1 = self.U.from_series(self.mock_series_arm_1)
        use_arm_2 = self.U.from_series(self.mock_series_arm_2)
        use_target = self.U.from_series(self.mock_series_target)

        E = EmbeddingCache(metadata=dict())
        
        self.assertEqual(E._targets_with_new_uses, set())
        self.assertEqual(E._cache, {})

        E.add_use(use=use_arm_1, embedding=torch.Tensor())
        self.assertEqual(E._targets_with_new_uses, {'arm'})
        # E._cache = {'arm': {'A': tensor([])}}
        self.assertEqual(list(E._cache['arm'].keys()), ['A'])

        E.add_use(use=use_arm_1, embedding=torch.Tensor())
        self.assertEqual(E._targets_with_new_uses, {'arm'})
        # E._cache = {'arm': {'A': tensor([]), 'D': tensor([])}}
        self.assertEqual(list(E._cache['arm'].keys()), ['A'])

        E.add_use(use=use_arm_2, embedding=torch.Tensor())
        E.add_use(use=use_target, embedding=torch.Tensor())
        self.assertEqual(E._targets_with_new_uses, {'arm', 'target'})
        self.assertEqual(list(E._cache.keys()), ['arm', 'target'])
        self.assertEqual(list(E._cache['arm'].keys()), ['A', 'D'])

        E.add_use(use=use_arm_1, embedding=torch.Tensor())
        self.assertEqual(E._targets_with_new_uses, {'arm', 'target'})
        self.assertEqual(list(E._cache.keys()), ['arm', 'target'])
        self.assertEqual(list(E._cache['arm'].keys()), ['A', 'D'])
    
    @patch.object(EmbeddingCache, 'load')
    def test_retrieve_when_load_is_None(self, mock_load):
        mock_load.return_value = None
        use_empty = self.U.from_series(self.mock_series_empty)
        E = EmbeddingCache(metadata=dict())
        retrieve_return = E.retrieve(use=use_empty)
        self.assertEqual(retrieve_return, None)
    
    @patch.object(EmbeddingCache, 'load')
    def test_retrieve_target_in__cache(self, mock_load):
        mock_load.return_value = {'A': torch.tensor([[1, 2, 3], [4, 5, 6]])}
        use_arm = self.U.from_series(self.mock_series_arm_1)
        E = EmbeddingCache(metadata=dict())
        E._cache = {'target': {'D': torch.tensor([[3, 2, 1], [6, 5, 4]])}}
        self.assertTrue(torch.equal(E.retrieve(use=use_arm), torch.tensor([[1, 2, 3], [4, 5, 6]])))

    def test_retrieve_target_not_in__cache(self):
        use_arm = self.U.from_series(self.mock_series_arm_2)
        E = EmbeddingCache(metadata=dict())
        E._cache = {'arm': {'D': torch.tensor([[3, 2, 1], [6, 5, 4]])}}
        self.assertTrue(torch.equal(E.retrieve(use=use_arm), torch.tensor([[3, 2, 1], [6, 5, 4]])))

    def test_load(self):
        mock_metadata_target = {'dataset.name': 'testwug_en_111',
                    'dataset.preprocessing': 'raw',
                    'contextual_embedder.pre_target_tokens': 0.0, 
                    'contextual_embedder.ckpt': 'mock_model_name',
                    'id': 'mock_id', 
                    'target': 'target'}
        E = EmbeddingCache(metadata=mock_metadata_target)
        self.assertEqual(E.load(target='arm'), None)
    
    @patch('src.wic.contextual_embedder.torch.load')
    def test_load_when_match_in__index(self, mock_torch_load):
        E = EmbeddingCache(metadata=self.mock_metadata)
        E._index = pd.json_normalize(self.mock_metadata)
        E._index_dir = Path('mock/index/dir')
        return_of_load = E.load(target='arm')

        self.assertTrue(mock_torch_load.called) # check torch.load() called
        self.assertIs(type(return_of_load), dict) # check the type of return of load()
    
    def test__ids_when_index_exist(self):
        E = EmbeddingCache(metadata=dict())
        E._index = pd.json_normalize(self.mock_metadata)
        self.assertEqual(E._ids(), {'mock_id'})

    def test__ids_when_index_is_none(self):
        E = EmbeddingCache(metadata=dict())
        E._index = None
        _ids_return = E._ids()
        self.assertRaises(AssertionError)
        self.assertEqual(_ids_return, set())
    
    def test_targets(self):
        E = EmbeddingCache(metadata=dict())
        E._cache = {'arm': {'A': 0.0}, 'target': {'target-F': 0.0}}
        self.assertEqual(E.targets(), {'arm', 'target'})

    @patch('src.wic.contextual_embedder.pd.DataFrame.to_parquet')
    @patch('src.wic.contextual_embedder.Path.iterdir')
    @patch('src.wic.contextual_embedder.pd.DataFrame.drop_duplicates')
    def test_clean(self, mock_drop_dup, mock_iterdir, mock_to_parquet):
        E = EmbeddingCache(metadata=dict())
        E.clean()
        self.assertTrue(mock_drop_dup.called)
        self.assertTrue(mock_iterdir.called)
        self.assertTrue(mock_to_parquet.called)
    
    def test_clean_drop_duplicates(self):
        mock_metadata_duplicates = {
            'dataset.name': ['testwug_en_111', 'testwug_en_111', 'testwug_en_111', 'testwug_en_111', 'testwug_en_111', 'testwug_en_110', 'testwug_en_111'],
            'dataset.preprocessing': ['raw', 'raw', 'raw', 'raw', 'lemmatization', 'raw', 'raw'],
            'contextual_embedder.pre_target_tokens': [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], 
            'contextual_embedder.ckpt': ['mock_model_name', 'mock_model_name', 'mock_model_name_2', 'mock_model_name', 'mock_model_name', 'mock_model_name', 'mock_model_name'],
            'id': ['mock_id_0', 'mock_id_1', 'mock_id_2', 'mock_id_3', 'mock_id_4', 'mock_id_5', 'mock_id_6'], 
            'target': ['arm', 'target', 'arm', 'arm', 'arm', 'arm', 'arm']
            }
        mock_metadata_id_after_drop = ['mock_id_1', 'mock_id_2', 'mock_id_3', 'mock_id_4', 'mock_id_5', 'mock_id_6']
        E = EmbeddingCache(metadata=mock_metadata_duplicates)
        E._index = pd.DataFrame(mock_metadata_duplicates)
        E.clean()
        self.assertEqual(mock_metadata_id_after_drop, E._index.id.tolist())
    
    def test_persist_remove_from__targets_with_new_uses(self):
        E = EmbeddingCache(metadata=dict())
        E._targets_with_new_uses = {'arm', 'target'}
        E._cache = {'arm': {'A': 0.0}, 'target': {'target-F': 0.0}}
        E.persist(target='arm')
        self.assertEqual({'target'}, E._targets_with_new_uses)
    
    def test_persist_remove_nothing_from__targets_with_new_uses(self):
        with self.assertRaises(KeyError):
            E = EmbeddingCache(metadata=dict())
            E._targets_with_new_uses = {'target'}
            E._cache = {'arm': {'A': 0.0}, 'target': {'target-F': 0.0}}
            E.persist(target='arm')
    
    def test_persist_target_not_in__cache(self):
        with self.assertRaises(KeyError):
            E = EmbeddingCache(metadata=dict())
            E._targets_with_new_uses = {'arm', 'target'}
            E._cache = {'target': {'target-F': 0.0}}
            E.persist(target='arm')

    def test_persist_concate_target_entry_into__index(self):
        E = EmbeddingCache(metadata=self.mock_metadata)
        E._targets_with_new_uses = {'arm', 'target'}
        E._index = pd.json_normalize(dict())
        E._cache = {'arm': {'A': 0.0}, 'target': {'target-F': 0.0}}
        E.persist(target='arm')
        new_entry = E._index.iloc[[1]]
        new_entry.pop('id')
        self.assertEqual(['testwug_en_111', 'raw', 0.0, 'mock_model_name', 'arm'], new_entry.values.flatten().tolist())

    @patch('src.wic.contextual_embedder.torch.save')
    @patch('src.wic.contextual_embedder.open')
    @patch('src.wic.contextual_embedder.pd.concat')
    def test_persist_concate_and_open_and_save_are_called(self, mock_concate, mock_open, mock_save):
        E = EmbeddingCache(metadata=dict())
        E._targets_with_new_uses = {'arm', 'target'}
        E._cache = {'arm': {'A': 0.0}, 'target': {'target-F': 0.0}}
        E.persist(target='arm')
        self.assertTrue(mock_concate.called)
        self.assertTrue(mock_open.called)
        self.assertTrue(mock_save.called)

    def test_persist_logger(self):
        with self.assertLogs() as captured:
            E = EmbeddingCache(metadata=dict())
            E._targets_with_new_uses = {'arm', 'target'}
            E._cache = {'arm': {'A': 0.0}, 'target': {'target-F': 0.0}}
            E.persist(target='arm')
        self.assertEqual(len(captured.records), 2) # check that there is two log messages
        self.assertEqual(captured.records[0].getMessage(), "Logged record of new embedding file")
        self.assertIn("Saved embeddings to disk as", captured.records[1].getMessage())

    def test_has_new_uses(self):
        E = EmbeddingCache(metadata=dict())
        E._targets_with_new_uses = {'arm'}
        self.assertTrue(E.has_new_uses(target='arm'))
    
    def test_has_no_new_uses(self):
        E = EmbeddingCache(metadata=dict())
        E._targets_with_new_uses = {'arm'}
        self.assertFalse(E.has_new_uses(target='target'))

if __name__ == '__main__':
    unittest.main()
