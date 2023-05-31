import sys
sys.path.insert(0, ".")

from src.preprocessing import Lemmatize, Toklem, Raw, Normalize
import yaml
import re
import pandas as pd
from logging import getLogger
import unittest.mock as mock

import unittest


class TestPreprocessingEn(unittest.TestCase):
    mock_data = {'lemma': 'afternoon_nn',
                 'identifier': 'mag_2005_419574.txt-173-18',
                 'context': 'I\'ve never seen anything like it, " my father said to us all at tea that afternoon. "',
                 'indexes_target_token': '73:82',
                 'context_tokenized': 'I \'ve never seen anything like it , " my father said to us all at tea that afternoon . "',
                 'indexes_target_token_tokenized': 18,
                 'context_lemmatized': 'i have never see anything like it , " my father say to us all at tea that afternoon . "',
                }

    mock_series = pd.Series(data=mock_data, index=['lemma', 'identifier', 'context', 'indexes_target_token', 'context_tokenized', 'indexes_target_token_tokenized', 'context_lemmatized'])

    translation_table = {" n't": "n't", " 've": "'ve"}

    def test_Lemmatize_spelling_normalization(self):
        L = Lemmatize(spelling_normalization=self.translation_table)
        L_series = L.fields_from_series(self.mock_series)

        mock_lemma = 'afternoon'
        mock_context_lemmatized = 'i have never see anything like it , " my father say to us all at tea that afternoon . "'
        mock_indexes_target_token_tokenized = 18
        mock_start = next(re.finditer(pattern=mock_lemma, string=mock_context_lemmatized))

        self.assertEqual(L_series['context'], mock_context_lemmatized)
        self.assertEqual(L_series['index'], mock_indexes_target_token_tokenized)

        context, start, end = L.preprocess(L_series['context'], L_series['index'])
        print('L', start)
        print('L', end)

        self.assertEqual(start, mock_start.start())
    
    def test_Raw(self):
        
        R = Raw(spelling_normalization=self.translation_table)
        R_series = R.fields_from_series(self.mock_series)
        mock_context_raw = 'I \'ve never seen anything like it, " my father said to us all at tea that afternoon. "'
        mock_start = 73
        mock_end = 82

        self.assertEqual(R_series['context'], mock_context_raw)
        self.assertEqual(R_series['start'], mock_start)
        self.assertEqual(R_series['end'], mock_end)
    
    def test_Normalize(self):
        N = Normalize(spelling_normalization=self.translation_table, default='context_tokenized')
        
        log = getLogger('src.preprocessing')
        with mock.patch.object(log, 'warn') as mock_warn:
            N_series = N.fields_from_series(self.mock_series)
            mock_warn.assert_called_once_with("(lemma=afternoon_nn, use=mag_2005_419574.txt-173-18) does not contain a pre-normalized context, context_tokenized will be used")

        mock_lemma = 'afternoon'

        mock_context_tokenized = 'I \'ve never seen anything like it , " my father said to us all at tea that afternoon . "'
        mock_indexes_target_token_tokenized = 18

        self.assertEqual(N_series['context'], mock_context_tokenized)
        self.assertEqual(N_series['index'], mock_indexes_target_token_tokenized)

        mock_preprocess_context = 'I\'ve never seen anything like it , " my father said to us all at tea that afternoon . "'
        mock_start = next(re.finditer(pattern=mock_lemma, string=mock_preprocess_context))

        context, start, end = N.preprocess(N_series['context'], N_series['index'])

        self.assertEqual(context, mock_preprocess_context)
        self.assertEqual(start, mock_start.start())
        self.assertEqual(end, mock_start.end() - 1)
    
    def test_character_indices_recalculation_one_after(self) -> None:
        context = "I do n't agree that it could 've won"
        target = "agree"
        span = next(re.finditer(pattern=target, string=context))
        preprocessor = Toklem(spelling_normalization=self.translation_table)
        _, start = preprocessor.normalize_spelling(
                                context, span.start()
                                )
        self.assertEqual(start, span.start() - 1)

    def test_character_indices_recalculation_several_after(self) -> None:
        context = "I do n't, I ca n't, I would n't agree that it could 've won"
        target = "agree"
        span = next(re.finditer(pattern=target, string=context))
        preprocessor = Toklem(spelling_normalization=self.translation_table)
        context_nor, start = preprocessor.normalize_spelling(context, span.start())
        self.assertEqual(start, span.start() - 3)


    def test_character_indices_recalculation_several_around(self) -> None:
        context = "I do n't, I can, I would n't agree that it could 've won"
        target = "can"
        span = next(re.finditer(pattern=target, string=context))
        preprocessor = Toklem(spelling_normalization=self.translation_table)
        _, start = preprocessor.normalize_spelling(context, span.start())
        self.assertEqual(start, span.start() - 1)


    def test_character_indices_recalculation_before(self) -> None:
        context = "I agree do n't that it could 've won"
        target = "agree"
        span = next(re.finditer(pattern=target, string=context))

        preprocessor = Toklem(spelling_normalization=self.translation_table)
        _, start = preprocessor.normalize_spelling(context, span.start())
        self.assertEqual(start, span.start())

    def test_character_indices_recalculation_introduce_blanks(self) -> None:
        context = "I agree don't that it could've won"
        target = "that"
        span = next(re.finditer(pattern=target, string=context))
        
        preprocessor = Toklem(spelling_normalization=self.translation_table)
        _, start = preprocessor.normalize_spelling(context, span.start())
        self.assertEqual(start, span.start() + 1)


    def test_toklem_preprocessing_longer(self) -> None:
        context = "He agrees that it could 've won"
        
        preprocessor = Toklem(spelling_normalization=self.translation_table)
        lemma = "agreess"
        new_context, start, end = preprocessor.preprocess(
            context=context, lemma=lemma, index=1
        )
        slice = new_context[start:end+1]
        self.assertEqual(new_context, "He agreess that it could've won")
        self.assertEqual(start, 3)
        self.assertEqual(end, 9)
        self.assertEqual(slice, lemma)

    def test_toklem_preprocessing_equal(self) -> None:
        context = "He agrees that it could 've won"
        
        preprocessor = Toklem(spelling_normalization=self.translation_table)
        lemma = "agrees"
        new_context, start, end = preprocessor.preprocess(
            context=context, lemma=lemma, index=1
        )
        slice = new_context[start:end+1]
        self.assertEqual(new_context, "He agrees that it could've won")
        self.assertEqual(slice, lemma)
        self.assertEqual(start, 3)
        self.assertEqual(end, 8)

    def test_toklem_preprocessing_shorter_no_sp_norm(self) -> None:
        context = "He agrees that it could 've won"
        preprocessor = Toklem(spelling_normalization=None)
        lemma = "agree"

        new_context, start, end = preprocessor.preprocess(
            context=context, lemma=lemma, index=1
        )
        slice = new_context[start:end+1]
        self.assertEqual(new_context, "He agree that it could 've won")
        self.assertEqual(start, 3)
        self.assertEqual(end, 7)
        self.assertEqual(slice, lemma)

    def test_real_case(self) -> None:
        context = "In the afternoons , tea and what appeared to be crumpets were set out near a patio ."
        preprocessor = Toklem(spelling_normalization=None)
        lemma = "afternoon"
        new_context, start, end = preprocessor.preprocess(
            context=context, lemma=lemma, index=2
        )
        slice = new_context[start:end+1]
        self.assertEqual(new_context, "In the afternoon , tea and what appeared to be crumpets were set out near a patio .")
        self.assertEqual(start, 7)
        self.assertEqual(end, 15)
        self.assertEqual(slice, lemma)

    def test_toklem_preprocessing_shorter(self) -> None:
        context = "He agrees that it could 've won"
        
        preprocessor = Toklem(spelling_normalization=self.translation_table)
        lemma = "agree"
        new_context, start, end = preprocessor.preprocess(
            context=context, lemma=lemma, index=1
        )

        slice = new_context[start:end+1]
        self.assertEqual(new_context, "He agree that it could've won")
        self.assertEqual(start, 3)
        self.assertEqual(end, 7)
        self.assertEqual(slice, lemma)

if __name__ == '__main__':
    unittest.main()
