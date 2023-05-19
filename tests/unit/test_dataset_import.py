import sys
sys.path.insert(0, ".")

import os

import unittest

class TestDataset(unittest.TestCase):

    def test_dataset_import(self):
        groupings = tuple(['1', '2'])
        type = 'dev'
        split = 'dev'
        exclude_annotators = []
        name = 'testwug_en'
        test_on = None
        cleaning = None
        path = 'testwug_en'
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

        datasetdirpath = os.path.abspath('wug')
        print(datasetdirpath)
        datasetpath = os.path.join(datasetdirpath, path)

        self.assertTrue(os.path.isdir(datasetpath), "The dataset is not imported.")
        

if __name__ == '__main__':
    unittest.main()
