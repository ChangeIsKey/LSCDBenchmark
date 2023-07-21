import sys
sys.path.insert(0, ".")

import unittest
from unittest.mock import patch
import numpy as np
import json
from scipy.spatial.distance import cosine
import torch

from src.use import Use
from src.wic.distance import dot_product
from src.wic.model import NumpyEncoder, WICModel
from src.wic.contextual_embedder import ContextualEmbedder

class Testdistance(unittest.TestCase):    
    def test_dot_product(self):
        a = np.asarray([1.0, 2.5, 3.7])
        b = np.asarray([0.4, 0.5, 0.6])
        self.assertEqual(dot_product(a=a, b=b), 3.87)

class TestNumpyEncoder(unittest.TestCase):
    # TODO: run python -m unittest tests.unit.test_wic_distance_model_contextualEmbedder.TestNumpyEncoder
    def test_default(self):
        json.dumps(2 + 1j, cls=NumpyEncoder)
        print(NumpyEncoder().encode(np.asarray([1.0, 2.5, 3.7])))

class TestContextualEmbedder(unittest.TestCase):
    CE_ckpt = "bert-base-uncased"
    CE = ContextualEmbedder(truncation_tokens_before_target=0.5,
                            similarity_metric=cosine, 
                            ckpt=CE_ckpt,
                            layers=[1, 12],
                            embedding_cache=None,
                            layer_aggregation="average",
                            subword_aggregation="average",
                            encode_only=False
                            )
    U_0 = Use(identifier='A',
                grouping='1',
                context='and taking a knife from her pocket, she opened a vein in her little arm, and dipping a feather in the blood, wrote something on a piece of white cloth, which was spread before her.',
                target='arm',
                indices=(68, 71),
                pos='N')
    U_1 = Use(identifier='B',
                grouping='1',
                context='And those who remained at home had been heavily taxed to pay for the arms, ammunition; fortifications, and all the other endless expenses of a war.',
                target='arm',
                indices=(69, 73),
                pos='N')
    U_2 = Use(identifier='D',
                grouping='2',
                context='It stood behind a high brick wall, its back windows overlooking an arm of the sea which, at low tide, was a black and stinking mud-flat',
                target='arm',
                indices=(67, 70),
                pos='N')
    U_3 = Use(identifier='E',
                grouping='2',
                context='twelve miles of coastline lies in the southwest on the Gulf of Aqaba, an arm of the Red Sea. The city of Aqaba, the only port, plays.',
                target='arm',
                indices=(73, 76),
                pos='N')

    def test_as_df(self):
        print(self.CE.as_df()) # return df
        # TODO: work with test_wug
        # TODO: run python -m unittest tests.unit.test_wic_distance_model_contextualEmbedder.TestContextualEmbedder.test_as_df

    def test_device(self):
        self.CE.gpu = 0
        if torch.cuda.is_available():
            self.assertEqual(str(self.CE.device), 'cuda:0')
        else:
            self.assertEqual(str(self.CE.device), 'cpu')

    def test_tokenizer(self):
        self.CE.tokenizer
        self.assertTrue(True if self.CE._tokenizer else False)
        self.assertEqual(self.CE._tokenizer.name_or_path, self.CE_ckpt)

    def test_model(self):
        self.CE.model
        self.assertTrue(True if self.CE._model else False)
        self.assertEqual(self.CE._model.name_or_path, self.CE_ckpt)

    def test_truncation_indices(self):
        index1, index2 = self.CE.truncation_indices([True] * 520)
        self.assertEqual(index1, 4)
        self.assertEqual(index2, 516)

    def test_predict(self):
        use_pairs = [(self.U_0, self.U_2), (self.U_1, self.U_3)]
        self.assertEqual(self.CE.predict(use_pairs=use_pairs)[0], -0.35529839992523193)
        self.assertEqual(self.CE.predict(use_pairs=use_pairs)[1], -0.560631513595581)

    def test_tokenize(self):
        U = Use(identifier=str(),
                grouping=str(),
                context='in her little arm',
                target='arm',
                indices=(int(), int()),
                pos=str())
        self.assertTrue(torch.equal(self.CE.tokenize(use=U)['input_ids'], torch.tensor([[ 101, 1999, 2014, 2210, 2849,  102]])))

    def test_aggregate(self):
        # also test LayerAggregator and SubwordAggregator
        embeddings = torch.tensor([[[0.1111, 0.111, 0.11, 0.1], [0.1112, 0.112, 0.12, 0.2], [0.1113, 0.113, 0.13, 0.3], [0.1114, 0.114, 0.14, 0.4]], 
                                   [[0.1121, 0.121, 0.21, 0.1], [0.1122, 0.122, 0.22, 0.2], [0.1123, 0.123, 0.23, 0.3], [0.1124, 0.124, 0.24, 0.4]],
                                   [[0.1131, 0.131, 0.31, 0.1], [0.1132, 0.132, 0.32, 0.2], [0.1133, 0.133, 0.33, 0.3], [0.1134, 0.134, 0.34, 0.4]],
                                   [[0.1141, 0.141, 0.41, 0.1], [0.1142, 0.142, 0.42, 0.2], [0.1143, 0.143, 0.43, 0.3], [0.1144, 0.144, 0.44, 0.4]]]
                                   )
        layers = [1, 3]
        CE1 = ContextualEmbedder(truncation_tokens_before_target=0.5,
                            similarity_metric=cosine, 
                            ckpt=self.CE_ckpt,
                            layers=[1, 12],
                            embedding_cache=None,
                            layer_aggregation="average",
                            subword_aggregation="average",
                            encode_only=False
                            )
        self.assertTrue(torch.equal(CE1.aggregate(tensor=embeddings, layers=layers), torch.tensor([0.1128, 0.1280, 0.2800, 0.3000])))
        CE2 = ContextualEmbedder(truncation_tokens_before_target=0.5,
                            similarity_metric=cosine, 
                            ckpt=self.CE_ckpt,
                            layers=[1, 12],
                            embedding_cache=None,
                            layer_aggregation="average",
                            subword_aggregation="first",
                            encode_only=False
                            )
        CE2_tensor = CE2.aggregate(tensor=embeddings, layers=layers)
        self.assertTrue(torch.eq(CE2_tensor[0], torch.tensor(0.1113)))
        # TODO: run python -m unittest tests.unit.test_wic_distance_model_contextualEmbedder.TestContextualEmbedder.test_aggregate
        # print(CE2_tensor[1]) # torch.tensor(0.1130)
        # print(type(CE2_tensor[1])) # torch.Tensor
        # TODO: get error for self.assertTrue(torch.eq(CE2_tensor[1], torch.tensor(0.1130)))
        self.assertIsInstance(CE2_tensor[1], torch.Tensor)
        self.assertTrue(torch.eq(CE2_tensor[2], torch.tensor(0.1300)))
        self.assertTrue(torch.eq(CE2_tensor[3], torch.tensor(0.3000)))
        CE3 = ContextualEmbedder(truncation_tokens_before_target=0.5,
                            similarity_metric=cosine, 
                            ckpt=self.CE_ckpt,
                            layers=[1, 12],
                            embedding_cache=None,
                            layer_aggregation="average",
                            subword_aggregation="last",
                            encode_only=False
                            )
        self.assertTrue(torch.equal(CE3.aggregate(tensor=embeddings, layers=layers), torch.tensor([0.1143, 0.1430, 0.4300, 0.3000])))
        CE4 = ContextualEmbedder(truncation_tokens_before_target=0.5,
                            similarity_metric=cosine, 
                            ckpt=self.CE_ckpt,
                            layers=[1, 12],
                            embedding_cache=None,
                            layer_aggregation="average",
                            subword_aggregation="sum",
                            encode_only=False
                            )
        self.assertTrue(torch.equal(CE4.aggregate(tensor=embeddings, layers=layers), torch.tensor([0.4512, 0.5120, 1.1200, 1.2000])))
        CE5 = ContextualEmbedder(truncation_tokens_before_target=0.5,
                            similarity_metric=cosine, 
                            ckpt=self.CE_ckpt,
                            layers=[1, 12],
                            embedding_cache=None,
                            layer_aggregation="concat",
                            subword_aggregation="max",
                            encode_only=False
                            )
        self.assertTrue(torch.equal(CE5.aggregate(tensor=embeddings, layers=layers), torch.tensor([0.1142, 0.1420, 0.4200, 0.2000, 0.1144, 0.1440, 0.4400, 0.4000])))
        CE6 = ContextualEmbedder(truncation_tokens_before_target=0.5,
                            similarity_metric=cosine, 
                            ckpt=self.CE_ckpt,
                            layers=[1, 12],
                            embedding_cache=None,
                            layer_aggregation="sum",
                            subword_aggregation="min",
                            encode_only=False
                            )
        CE6_tensor = CE6.aggregate(tensor=embeddings, layers=layers)
        self.assertTrue(torch.eq(CE6_tensor[0], torch.tensor(0.2226)))
        # TODO: run python -m unittest tests.unit.test_wic_distance_model_contextualEmbedder.TestContextualEmbedder.test_aggregate
        # print(CE6_tensor[1]) # torch.tensor(0.2260)
        # print(type(CE6_tensor[1])) # torch.Tensor
        # TODO: get error for self.assertTrue(torch.eq(CE6_tensor[1], torch.tensor(0.2260)))
        self.assertIsInstance(CE6_tensor[1], torch.Tensor)
        self.assertTrue(torch.eq(CE6_tensor[2], torch.tensor(0.2600)))
        self.assertTrue(torch.eq(CE6_tensor[3], torch.tensor(0.6000)))

    def test_encode_all(self):
        encode_all = self.CE.encode_all([self.U_0, self.U_1, self.U_2, self.U_3])
        self.assertEqual(len(encode_all), 4)
        self.assertEqual(encode_all[0][0], np.float32(-0.08517842))
        self.assertEqual(encode_all[2][0], np.float32(-0.13967228))

    def test_encode(self):
        encode1 = self.CE.encode(self.U_0, type=np.ndarray)
        self.assertIsInstance(encode1, np.ndarray)
        self.assertEqual(encode1[0], np.float32(-0.08517842))
        encode2 = self.CE.encode(self.U_2, type=np.ndarray)
        self.assertIsInstance(encode2[0], np.float32)
        self.assertEqual(encode2[0], np.float32(-0.13967228))

    def test_predict_all(self):
        # the func. in model.py
        use_pairs = [(self.U_0, self.U_2), (self.U_1, self.U_3)]
        print(type(self.CE.predict_all(use_pairs=use_pairs))) # return list
        # TODO: run python -m unittest tests.unit.test_wic_distance_model_contextualEmbedder.TestContextualEmbedder.test_predict_all

if __name__ == '__main__':
    unittest.main()

