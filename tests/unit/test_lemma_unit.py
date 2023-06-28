import sys
sys.path.insert(0, ".")
import os

import unittest
from unittest.mock import patch, create_autospec

from pydantic import DirectoryPath
import pathlib
import pandas as pd
import csv


from src.lemma import Lemma, RandomSampling
from src.preprocessing import Lemmatize, Raw
from src.dataset import Dataset
import src


class TestLemma(unittest.TestCase):    
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

    mock_testwug_path = pathlib.Path('wug/testwug_en_111/data/afternoon_nn')
    mock_uses_data = {'lemma': 'afternoon_nn',
                      'grouping': '2',
                      'identifier': 'mag_2005_419574.txt-173-18',
                      'context': 'I\'ve never seen anything like it, " my father said to us all at tea that afternoon. "',
                      'indexes_target_token': '73:82',
                      'context_tokenized': 'I \'ve never seen anything like it , " my father said to us all at tea that afternoon . "',
                      'indexes_target_token_tokenized': '18',
                      'context_lemmatized': 'i have never see anything like it , " my father say to us all at tea that afternoon . "'
                      } # from row index 187
        
    Lem = Lemmatize(spelling_normalization=None)
    R = Raw(spelling_normalization=None)
    L_Lem = Lemma(groupings=('1', '2'), 
              path=mock_testwug_path,
              preprocessing=Lem)
    L_R = Lemma(groupings=('1', '2'), 
              path=mock_testwug_path,
              preprocessing=R)
    L_R_arm = Lemma(groupings=('1', '2'), 
              path=pathlib.Path('wug/testwug_en_111/data/arm'),
              preprocessing=R)

    def test_name(self):
        self.assertEqual(self.L_Lem.name, 'afternoon_nn')

    def test_differernt_preprocess_in_uses_df(self):
        # preprocess correctly when call from uses_df
        # lemmatize
        self.assertEqual(self.L_Lem.uses_df.context_preprocessed.iloc[187], self.mock_uses_data['context_lemmatized'])
        # raw
        self.assertEqual(self.L_R.uses_df.context_preprocessed.iloc[187], self.mock_uses_data['context'])    

    def test_annotated_pairs_df(self):
        mock_judgments_data = {'identifier1': ['mag_1986_492553.txt-134-22', 'fic_1992_40272.txt-627-18'],
                               'identifier2': ['fic_1831_7210.txt-510-4', 'fic_1846_7036.txt-2381-16']
                               } # from row 0 and 1
        # test if annotated_pairs_df return the correct data
        self.assertEqual(self.L_Lem.annotated_pairs_df.identifier1.iloc[0], mock_judgments_data['identifier1'][0])
        self.assertEqual(self.L_Lem.annotated_pairs_df.identifier2.iloc[1], mock_judgments_data['identifier2'][1])

    def test_augmented_annotated_pairs_df(self):
        mock_uses_data2 = {'grouping': '2',
                           'identifier': 'mag_1980_491790.txt-342-2',
                           } # from uses row index 106
        
        self.assertEqual(self.L_Lem.augmented_annotated_pairs_df.identifier1.iloc[184], self.mock_uses_data['identifier'])
        self.assertEqual(self.L_Lem.augmented_annotated_pairs_df.identifier2.iloc[184], mock_uses_data2['identifier'])
        self.assertEqual(self.L_Lem.augmented_annotated_pairs_df.grouping_x.iloc[184], mock_uses_data2['grouping'])
        self.assertEqual(list(self.L_Lem.augmented_annotated_pairs_df.columns), ["identifier1", "identifier2", "grouping_x", "grouping_y"])

    def test_useid_to_grouping(self):
        dict_useid_to_grouping = self.L_Lem.useid_to_grouping()
        self.assertEqual(dict_useid_to_grouping['fic_1855_2965.txt-1882-18'], '1')
        self.assertEqual(dict_useid_to_grouping['fic_1999_29239.txt-65-11'], '2')

    def test_grouping_to_useid(self):
        dict_grouping_to_useid = self.L_Lem.grouping_to_useid()
        self.assertTrue('fic_1855_2965.txt-1882-18' in dict_grouping_to_useid['1'])
        self.assertFalse('fic_1855_2965.txt-1882-18' in dict_grouping_to_useid['2'])
        self.assertTrue('fic_1999_29239.txt-65-11' in dict_grouping_to_useid['2'])
        self.assertFalse('fic_1999_29239.txt-65-11' in dict_grouping_to_useid['1'])

    def test_split_uses(self):
        # self.mock_uses_data['identifier'] is in group 2 and row index 86
        self.assertTrue(self.L_Lem.split_uses("COMPARE")[1][85], self.mock_uses_data['identifier'])
        # test if data send to correct position in split_uses("ALL")
        # also test ._split_compare_uses()
        self.assertTrue(self.L_Lem.split_uses("COMPARE")[0][0], self.L_Lem.split_uses("ALL")[0][0])
        self.assertTrue(self.L_Lem.split_uses("COMPARE")[1][85], self.L_Lem.split_uses("ALL")[1][85])
        # also test ._split_earlier_uses()
        self.assertTrue(self.L_Lem.split_uses("EARLIER")[0][0], self.L_Lem.split_uses("ALL")[0][100])
        self.assertTrue(self.L_Lem.split_uses("EARLIER")[1][85], self.L_Lem.split_uses("ALL")[1][185])
        # also test ._split_later_uses()
        self.assertTrue(self.L_Lem.split_uses("LATER")[0][0], self.L_Lem.split_uses("ALL")[0][200])
        self.assertTrue(self.L_Lem.split_uses("LATER")[1][85], self.L_Lem.split_uses("ALL")[1][285])

    def test_get_uses(self):
        self.assertTrue(self.L_Lem.get_uses()[185].identifier, self.mock_uses_data['identifier'])

    def test_use_pairs(self):
        RS = RandomSampling(n=2, replace=False)
        
        # 'EARLIER', 'annotated': both Uses in the pair belongs to groups 1
        self.assertEqual(len(self.L_R_arm.use_pairs('EARLIER', 'annotated')), 3)
        self.assertEqual(self.L_R_arm.use_pairs('EARLIER', 'annotated')[0][0].identifier, 'A') 
        self.assertEqual(self.L_R_arm.use_pairs('EARLIER', 'annotated')[0][1].identifier, 'B')
        # 'EARLIER', 'all': the first Use in the pair belongs to groups 1, the second belongs to groups 2
        self.assertEqual(len(self.L_R_arm.use_pairs('EARLIER', 'all')), 9)
        self.assertEqual(self.L_R_arm.use_pairs('EARLIER', 'all')[4][0].identifier, 'B') 
        self.assertEqual(self.L_R_arm.use_pairs('EARLIER', 'all')[4][0].identifier, self.L_R_arm.use_pairs('EARLIER', 'all')[4][1].identifier) # only keep the Use in group 1

        self.assertEqual(len(self.L_Lem.use_pairs('COMPARE', RS)), RS.n)
        self.assertIsInstance(self.L_Lem.use_pairs('COMPARE', RS)[0], tuple)
        self.assertIsInstance(self.L_Lem.use_pairs('COMPARE', RS)[0][0], src.use.Use)


    def test_with_use_pairs_csv(self):
        parent_dir = 'mock_wug'
        os.makedirs(parent_dir, exist_ok=True) # create a directory recursively
        mock_testwug_path = pathlib.Path(parent_dir)

        path_to_use_pairs_csv = os.path.join(parent_dir, 'use_pairs.csv')
        mock_use_pairs_data = pd.DataFrame({'lemma': ['arm', 'arm', 'arm', 'arm'],
                        'identifier1': ['A', 'B', 'C', 'E'],
                        'identifier2': ['B', 'F', 'F', 'F'],
                        'annotator': ['annotator', 'annotator', 'annotator', 'annotator']
                        })
        mock_use_pairs_data.to_csv(path_to_use_pairs_csv, index=False, sep='\t', quoting=csv.QUOTE_NONE) # use quoting=csv.QUOTE_NONE to solve when the df includes quote and pass to df.to_csv 
        
        path_to_uses_csv = os.path.join(parent_dir, 'uses.csv')
        mock_uses_data = pd.DataFrame({'lemma': ['arm', 'arm', 'arm', 'arm', 'arm'],
                                       'grouping': ['1', '1', '1', '2', '2'],
                                       'identifier': ['A', 'B', 'C', 'E', 'F'],
                                       'pos': ['N', 'N', 'N', 'N', 'N'],
                                       'context': ['and taking a knife from her pocket, she opened a vein in her little arm, and dipping a feather in the blood, wrote something on a piece of white cloth, which was spread before her.',
                                                   'And those who remained at home had been heavily taxed to pay for the arms, ammunition; fortifications, and all the other endless expenses of a war.', 
                                                   'and though he saw her within reach of his arm, yet the light of her eyes seemed as far off as that of a',
                                                   'twelve miles of coastline lies in the southwest on the Gulf of Aqaba, an arm of the Red Sea. The city of Aqaba, the only port, plays.', 
                                                   'when the disembodied arm of the Statue of Liberty jets spectacularly out of the sandy beach.'],
                                       'indexes_target_token':  ['68:71', '69:73', '42:45', '73:76', '21:24']
                                       })			
        mock_uses_data.to_csv(path_to_uses_csv, index=False, sep='\t', quoting=csv.QUOTE_NONE) # use quoting=csv.QUOTE_NONE to solve when the df includes quote and pass to df.to_csv 
        
        L = Lemma(groupings=('1', '2'), 
                    path=mock_testwug_path,
                    preprocessing=self.R)
        
        # test if predefined_use_pairs_df return the correct data
        self.assertEqual(L.predefined_use_pairs_df.identifier1.iloc[0], mock_use_pairs_data['identifier1'][0])
        self.assertEqual(L.predefined_use_pairs_df.identifier2.iloc[1], mock_use_pairs_data['identifier2'][1])

        # test if augmented_predefined_use_pairs_df return the correct data
        self.assertEqual(L.augmented_predefined_use_pairs_df.identifier1.iloc[0], mock_use_pairs_data['identifier1'][0])
        self.assertEqual(L.augmented_predefined_use_pairs_df.identifier2.iloc[2], mock_use_pairs_data['identifier2'][2])
        self.assertEqual(L.augmented_predefined_use_pairs_df.grouping_y.iloc[1], mock_uses_data['grouping'][4])
        self.assertEqual(list(L.augmented_predefined_use_pairs_df.columns), ["identifier1", "identifier2", "grouping_x", "grouping_y"])

        # test if _split_augmented_uses return the correct data
        # when the pair correspondingly belongs to different groups, e.g. pair: B - F and pair: C - F
        self.assertEqual(L._split_augmented_uses('COMPARE', L.augmented_predefined_use_pairs_df), (['B', 'C'], ['F', 'F']))
        # when the pair belongs to group 1, e.g. pair: A - B
        self.assertEqual(L._split_augmented_uses('EARLIER', L.augmented_predefined_use_pairs_df), (['A'], ['B']))
        # when the pair belongs to group 1, e.g. pair: E - F
        self.assertEqual(L._split_augmented_uses('LATER', L.augmented_predefined_use_pairs_df), (['E'], ['F']))
        self.assertEqual(L._split_augmented_uses('ALL', L.augmented_predefined_use_pairs_df), (['B', 'C', 'A', 'E'], ['F', 'F', 'B', 'F']))
        
        # test if sample as predefined in use_pairs 
        self.assertEqual(L.use_pairs('COMPARE', 'predefined')[1][0].identifier, mock_use_pairs_data['identifier1'][2]) # C
        self.assertEqual(L.use_pairs('COMPARE', 'predefined')[1][1].identifier, mock_use_pairs_data['identifier2'][2]) # F

        os.remove(path_to_use_pairs_csv)
        os.remove(path_to_uses_csv)
        os.rmdir(parent_dir)

if __name__ == '__main__':
    unittest.main()  
