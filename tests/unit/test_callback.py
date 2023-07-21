import sys
sys.path.insert(0, ".")
import json
import os
import csv
from omegaconf import DictConfig
from hydra.core.utils import JobReturn, JobStatus
import shutil

import unittest

from src.callback import *
from src.utils import utils

class TestLogJobReturnCallback(unittest.TestCase):

    LJRC = LogJobReturnCallback()

    def test_on_job_end(self):
        JR = JobReturn
        with self.assertLogs() as captured:
            self.LJRC.on_job_end(config=DictConfig, job_return=JR)
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "Status unknown. This should never happen.")
        
        JR_com = JobReturn
        JR_com.status = JobStatus.COMPLETED
        JR_com.return_value = 0.123

        with self.assertLogs() as captured:
            self.LJRC.on_job_end(config=DictConfig, job_return=JR_com)
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "Succeeded with return value: 0.123")

        JR_fail = JobReturn
        JR_fail.status = JobStatus.FAILED
        JR_fail.return_value = 0.456

        with self.assertLogs() as captured:
            self.LJRC.on_job_end(config=DictConfig, job_return=JR_fail)
        self.assertEqual(len(captured.records), 1)

class TestExperiment(unittest.TestCase):

    mock_path_RE = Path('mock_outputs/2023-06-05/16-00-03')
    RE = RunExperiment(path=mock_path_RE)
    mock_path_RE_hydra = mock_path_RE / ".hydra"

    mock_path_ME = Path('mock_multirun/2023-06-05/16-00-03/0')
    ME = MultirunExperiment(path=mock_path_ME)
    
    def test_parse_timestamp(self):
        self.assertEqual(str(self.RE.timestamp), '2023-06-05 16:00:03')
        self.assertEqual(str(self.ME.timestamp), '2023-06-05 16:00:03')
    
    def test_process(self):
        # test get_score()
        write_mock_json(self.mock_path_RE)
        self.assertEqual(self.RE.get_score(), 0.5)

        # test get_config()
        dict_yaml = {"colors": ['red', 'yellow'],
                      "numbers": [0.3, 0.6]}
        write_mock_yaml(self.mock_path_RE_hydra)
        self.assertDictEqual(self.RE.get_config(), dict_yaml)

        # test get_n_targets()
        write_mock_csv(self.mock_path_RE)
        self.assertEqual(self.RE.get_n_targets(), 3)

        # test process()
        self.assertDictEqual(self.RE.process(), 
                             {'time': '2023-06-05 16:00:03', 
                              'score': 0.5, 
                              'n_targets': 3, 
                              'colors': ['red', 'yellow'], 
                              'numbers': [0.3, 0.6]}
                            )
        remove_dirs_and_files()

class TestResultCollector(unittest.TestCase):
    mock_path_RE = Path('mock_outputs/2023-06-05/16-00-03')
    mock_path_RE_hydra = mock_path_RE / ".hydra"

    mock_path_ME = Path('mock_multirun/2023-06-05/16-00-03/0')
    mock_path_ME_hydra = mock_path_ME / ".hydra"

    mock_path_results = Path('mock_results')

    RC = ResultCollector()
    RC.OUTPUTS_PATH = utils.path('mock_outputs')
    RC.MULTIRUN_PATH = utils.path('mock_multirun')
    RC.path = mock_path_results / "results"
    

    def test_write_results(self):
        self.RC.results = [{'instance': 'apple',
                            'prediction': 0.2,
                            'label': 0.7},
                           {'instance': 'apple',
                            'prediction': 0.2,
                            'label': 0.8}]
        self.RC.write_results()
        self.assertTrue(os.path.isfile("mock_results/results.json"))
        self.assertTrue(os.path.isfile("mock_results/results.csv"))
        with open('mock_results/results.csv') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for i, row in enumerate(reader):
                self.assertEqual(row['instance'], self.RC.results[i]['instance'])
                self.assertEqual(float(row['prediction']), self.RC.results[i]['prediction'])
        with open('mock_results/results.json') as f:
            json_data = json.load(f)
            for i, row in enumerate(json_data):
                self.assertDictEqual(row, self.RC.results[i])
        remove_dirs_and_files()

    def test_on_run_end(self):
        write_mock_json(self.mock_path_RE)
        write_mock_yaml(self.mock_path_RE_hydra)
        write_mock_csv(self.mock_path_RE)

        self.RC.on_run_end(config=DictConfig)
        # self.RC.results will be updated to be the same as results.json and results.csv
        self.assertTrue(os.path.isfile("mock_results/results.json"))
        self.assertTrue(os.path.isfile("mock_results/results.csv"))
        with open('mock_results/results.csv') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for i, row in enumerate(reader):
                self.assertEqual(row['time'], self.RC.results[i]['time'])
                self.assertEqual(float(row['score']), self.RC.results[i]['score'])
                self.assertEqual(int(row['n_targets']), self.RC.results[i]['n_targets'])
                self.assertEqual(row['colors'], str(self.RC.results[i]['colors']))
                self.assertEqual(row['numbers'], str(self.RC.results[i]['numbers']))
        with open('mock_results/results.json') as f:
            json_data = json.load(f)
            for i, row in enumerate(json_data):
                self.assertEqual(row['time'], self.RC.results[i]['time'])
                self.assertEqual(float(row['score']), self.RC.results[i]['score'])
                self.assertEqual(int(row['n_targets']), self.RC.results[i]['n_targets'])
                self.assertEqual(row['colors'], self.RC.results[i]['colors'])
                self.assertEqual(row['numbers'], self.RC.results[i]['numbers'])
        remove_dirs_and_files()

    def test_on_multirun_end(self):
        write_mock_json(self.mock_path_ME)
        write_mock_yaml(self.mock_path_ME_hydra)
        write_mock_csv(self.mock_path_ME)

        self.RC.on_multirun_end(config=DictConfig)
        # self.RC.results will be updated to be the same as results.json and results.csv
        self.assertTrue(os.path.isfile("mock_results/results.json"))
        self.assertTrue(os.path.isfile("mock_results/results.csv"))
        with open('mock_results/results.csv') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for i, row in enumerate(reader):
                self.assertEqual(row['time'], self.RC.results[i]['time'])
                self.assertEqual(float(row['score']), self.RC.results[i]['score'])
                self.assertEqual(int(row['n_targets']), self.RC.results[i]['n_targets'])
                self.assertEqual(row['colors'], str(self.RC.results[i]['colors']))
                self.assertEqual(row['numbers'], str(self.RC.results[i]['numbers']))
        with open('mock_results/results.json') as f:
            json_data = json.load(f)
            for i, row in enumerate(json_data):
                self.assertEqual(row['time'], self.RC.results[i]['time'])
                self.assertEqual(float(row['score']), self.RC.results[i]['score'])
                self.assertEqual(int(row['n_targets']), self.RC.results[i]['n_targets'])
                self.assertEqual(row['colors'], self.RC.results[i]['colors'])
                self.assertEqual(row['numbers'], self.RC.results[i]['numbers'])
        remove_dirs_and_files()

def write_mock_json(path):
    dict_score = {"score": 0.5, "metric": "spearmanr"}
    os.makedirs(path, exist_ok=True)
    path_json = path / "result.json"
    with open(path_json, "w") as outfile:
        json.dump(dict_score, outfile)

def write_mock_yaml(path):
    dict_yaml = {"colors": ['red', 'yellow'],
                      "numbers": [0.3, 0.6]}
    os.makedirs(path, exist_ok=True)
    path_yaml = path / "config.yaml"
    with open(path_yaml, 'w') as file:
        yaml.dump(dict_yaml, file)

def write_mock_csv(path):
    fieldnames = ['instance', 'prediction', 'label']
    dict_csv = [{'instance': 'apple',
                    'prediction': 0.2,
                    'label': 0.7},
                {'instance': 'apple',
                    'prediction': 0.2,
                    'label': 0.8},
                {'instance': 'apple',
                    'prediction': 0.3,
                    'label': 0.9}]
    path_csv =  path / "predictions.csv"
    with open(path_csv, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_csv)

def remove_dirs_and_files():
    # results/
    if os.path.exists("mock_results"):
        shutil.rmtree("mock_results")
    # multirun/
    if os.path.exists("mock_multirun"):
        shutil.rmtree("mock_multirun")
    # outputs
    if os.path.exists("mock_outputs"):
        shutil.rmtree("mock_outputs")

if __name__ == '__main__':
    unittest.main()


