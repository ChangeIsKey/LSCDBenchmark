import sys
sys.path.insert(0, ".")

import os
from unittest.mock import MagicMock, patch, Mock
import unittest

class TestDataset(unittest.TestCase):
    
    def test_creat_path(self):
        groupings = tuple(['1', '2'])
        type = 'dev'
        split = 'dev'
        exclude_annotators = []
        name = 'testwug_en_111'
        test_on = None
        cleaning = None
        D_path = 'testwug_en_111'
        url = 'https://zenodo.org/record/7946753/files/testwug_en.zip'

        import src.dataset
        D = src.dataset.Dataset(path=D_path, 
                    groupings=groupings, 
                    type=type,
                    split=split,
                    exclude_annotators=exclude_annotators,
                    name=name,
                    test_on=test_on,
                    cleaning=cleaning,
                    url=url)

        t_dirpath = os.path.abspath('wug')
        t_path = os.path.join(t_dirpath, D_path)

        self.assertEqual(str(D.data_dir), t_dirpath)
        self.assertEqual(str(D.relative_path), D_path)
        self.assertEqual(str(D.absolute_path), t_path)

    def test_url_is_None(self):
        
        with self.assertRaises(AssertionError):
            groupings = tuple(['1', '2'])
            type = 'dev'
            split = 'dev'
            exclude_annotators = []
            name = 'nordiachange_1'
            test_on = None
            cleaning = None
            path = 'nor_dia_change/subset1'
            url = None
            import src.dataset
            D = src.dataset.Dataset(path=path, 
                    groupings=groupings, 
                    type=type,
                    split=split,
                    exclude_annotators=exclude_annotators,
                    name=name,
                    test_on=test_on,
                    cleaning=cleaning,
                    url=url)
            D._Dataset__download_from_git()

    '''(error)
    def test__download_from_git(self):
        
        groupings = tuple(['1', '2'])
        type = 'dev'
        split = 'dev'
        exclude_annotators = []
        name = 'nordiachange_1'
        test_on = None
        cleaning = None
        D_path = 'nor_dia_change/subset1'
        t_path_parts_0 = 'nor_dia_change'
        url = 'https://github.com/ltgoslo/nor_dia_change.git'

        import src.dataset
        D = src.dataset.Dataset(path=D_path, 
                    groupings=groupings, 
                    type=type,
                    split=split,
                    exclude_annotators=exclude_annotators,
                    name=name,
                    test_on=test_on,
                    cleaning=cleaning,
                    url=url)

        # test for valid path
        D_to_path = str(D.data_dir) + '/' + str(D.relative_path.parts[0])

        t_dirpath = os.path.abspath('wug')
        t_to_path = t_dirpath + '/' + t_path_parts_0
        
        self.assertEqual(str(D.relative_path.parts[0]), t_path_parts_0)
        self.assertEqual(t_to_path, D_to_path)

        # test for downloading
        # D._Dataset__download_from_git()
        
        ## cmdline: git clone -v -- https://github.com/ltgoslo/nor_dia_change.git /mount/arbeitsdaten20/projekte/cik/users/kuan-yu/LSCDBenchmark/wug/nor_dia_change
        ##   stderr: 'fatal: destination path '/mount/arbeitsdaten20/projekte/cik/users/kuan-yu/LSCDBenchmark/wug/nor_dia_change' already exists and is not an empty directory.

        t_path = os.path.join(t_dirpath, D_path)

        self.assertTrue(os.path.isdir(t_path), "The dataset is not imported.")'''

    def test__download_zip(self):
        groupings = tuple(['1', '2'])
        type = 'dev'
        split = 'dev'
        exclude_annotators = []
        name = 'testwug_en_111'
        test_on = None
        cleaning = None
        path = 'testwug_en_111'
        url = 'https://zenodo.org/record/7946753/files/testwug_en.zip'

        import src.dataset
        D = src.dataset.Dataset(path=path, 
                    groupings=groupings, 
                    type=type,
                    split=split,
                    exclude_annotators=exclude_annotators,
                    name=name,
                    test_on=test_on,
                    cleaning=cleaning,
                    url=url)
        D._Dataset__download_zip()

        t_dirpath = os.path.abspath('wug')
        t_path = os.path.join(t_dirpath, path)
        t_path_with_suffix = t_path + '.zip'

        self.assertTrue(os.path.isdir(t_dirpath), "The directory does not exist.")
        self.assertEqual(str(D.absolute_path.with_suffix(".zip")), t_path_with_suffix)

        datasetdirpath = os.path.abspath('wug')

        datasetpath = os.path.join(datasetdirpath, path)
        dataset_zip_path = os.path.join(datasetdirpath, 'testwug_en.zip')

        self.assertFalse(os.path.isdir(dataset_zip_path), "The dataset is not unzipped.")
        self.assertTrue(os.path.isdir(datasetpath), "The dataset is not imported.")
        

    '''def test_filter_lemmas(self):
        pass

    def test_lemmas_proprocessing_none(self):
        pass
        # TODO: preprocessing = None

        # import src.dataset
        # D = src.dataset.Dataset(preprocessing=preprocessing)
        
        # self.assertRaises() : TypeError

    def test_lemma_to_load(self):
        pass
    
    def test_lemma_create_lemmas_list(self):
        pass
    '''

if __name__ == '__main__':
    unittest.main()
