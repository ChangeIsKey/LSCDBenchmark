import sys
sys.path.insert(0, ".")

import os

from unittest.mock import patch, Mock
import unittest

from src.dataset import Dataset

class TestDataset(unittest.TestCase):
    @patch.object(Dataset, 'relative_path')
    def test_relative_path(self, mock_relative_path):
        mock_relative_path.return_value = 'path'
        self.assertEqual(Dataset.relative_path(), 'path')
    
    @patch.object(Dataset, 'absolute_path')
    def test_absolute_path(self, mock_absolute_path):
        mock_absolute_path.return_value = 'path'
        self.assertEqual(Dataset.absolute_path(), 'path')

    @patch('src.utils.utils.path')
    @patch.object(Dataset, 'data_dir')
    @patch('src.dataset.os.getenv')
    def test_data_dir(self, mock_getenv, mock_data_dir, mock_path):
        mock_getenv.return_value = None
        mock_data_dir.return_value = 'wug'
        self.assertEqual(Dataset.data_dir(), 'wug')

    @patch('src.dataset.Dataset')
    def test_url_None(self, mock_Dataset):  
        mock_Dataset.url.return_value = None
        self.assertRaises(AssertionError)
    
    @patch.object(Dataset, '_Dataset__download_zip')
    @patch.object(Dataset, '_Dataset__download_from_git')
    @patch.object(Dataset, '_Dataset__download')
    def test__download(self, mock_download, ock_download_from_git, mock_download_zip):
        # Dataset._Dataset__download()
        # self.assertTrue(mock_download_from_git.called)
        # self.assertTrue(mock_download_zip.called)
        mock_download()
        mock_download.assert_called()

    '''@patch('src.dataset.Repo')
    def test__download_from_git(self, mock_repo):
        Dataset.url = Mock(return_value='url.git')
        Dataset._Dataset__download_from_git()
        mock_repo.assert_called()'''

    '''
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

        D = Dataset(path=D_path, 
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

        D = Dataset(path=path, 
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

        # D = Dataset(preprocessing=preprocessing)
        
        # self.assertRaises() : TypeError

    def test_lemma_to_load(self):
        pass
    
    def test_lemma_create_lemmas_list(self):
        pass
    '''

if __name__ == '__main__':
    unittest.main()
