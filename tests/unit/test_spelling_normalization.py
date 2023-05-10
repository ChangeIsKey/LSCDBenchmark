import sys
sys.path.append(r'../..')

from src.preprocessing import Lemmatize, Toklem
import yaml
import re

import unittest
import pytest


class TestSpellingNormalizationToklem(unittest.TestCase):
    """12 unit tests for each function in Toklem in spelling normalization.
    """    
    def test_character_indices_recalculation_one_after(self) -> None:
        context = "I do n't agree that it could 've won"
        target = "agree"
        span = next(re.finditer(pattern=target, string=context))
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Toklem(spelling_normalization=translation_table)
        _, start = preprocessor.normalize_spelling(
                                context, span.start()
                                )
        self.assertEqual(start, span.start() - 1)


    def test_character_indices_recalculation_several_after(self) -> None:
        context = "I do n't, I ca n't, I would n't agree that it could 've won"
        target = "agree"
        span = next(re.finditer(pattern=target, string=context))
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Toklem(spelling_normalization=translation_table)
        _, start = preprocessor.normalize_spelling(context, span.start())
        self.assertEqual(start, span.start() - 3)


    def test_character_indices_recalculation_several_around(self) -> None:
        context = "I do n't, I can, I would n't agree that it could 've won"
        target = "can"
        span = next(re.finditer(pattern=target, string=context))
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Toklem(spelling_normalization=translation_table)
        _, start = preprocessor.normalize_spelling(context, span.start())
        self.assertEqual(start, span.start() - 1)


    def test_character_indices_recalculation_before(self) -> None:
        context = "I agree do n't that it could 've won"
        target = "agree"
        span = next(re.finditer(pattern=target, string=context))
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Toklem(spelling_normalization=translation_table)
        _, start = preprocessor.normalize_spelling(context, span.start())
        self.assertEqual(start, span.start())

    def test_character_indices_recalculation_introduce_blanks(self) -> None:
        context = "I agree don't that it could've won"
        target = "that"
        span = next(re.finditer(pattern=target, string=context))
        translation_table = {"n't": " n't", "'ve": " 've"}
        preprocessor = Toklem(spelling_normalization=translation_table)
        _, start = preprocessor.normalize_spelling(context, span.start())
        self.assertEqual(start, span.start() + 1)


    def test_toklem_preprocessing_longer(self) -> None:
        context = "He agrees that it could 've won"
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Toklem(spelling_normalization=translation_table)
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
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Toklem(spelling_normalization=translation_table)
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
        

    def test_real_case_german(self) -> None:
        context = "Aufgeſang fuͤr Stoll , welches Docen durch ein Beiſpiel von 1515 rechtfertigt ( Note 11. ſeiner Abh. ) waͤre eine ſehr paſſende Benennung und entſpraͤche dem Abgeſang ."
        with open("../../conf/dataset/spelling_normalization/german.yaml", mode="r", encoding="utf8") as f:
            spelling_normalization = yaml.safe_load(f)
        preprocessor = Toklem(spelling_normalization=spelling_normalization)
        lemma = "Abgesang"
        new_context, start, end = preprocessor.preprocess(
            context=context, lemma=lemma, index=26
        )
        slice = new_context[start:end+1]
        self.assertEqual(slice, lemma)
        self.assertEqual(start, 155)
        self.assertEqual(end, 162)

    def test_real_case_german_2(self) -> None:
        context = "Seltner werden die Lieder , wo es hoͤher ſteigt , doch kenne ich zwei mit ſiebenreimigen Stollen , naͤmlich eins des Nifen 1. 23 . ( ſeht an die heide ꝛc. ) , wo der Abgeſang 9 , der ganze Ton alſo 23 Reime zaͤhlt , und eines von Canzler 2. 244. ( helfent mir ꝛc. ) , wo im Abgeſang 14 , im Ganzen 28 Reime ſtecken . Das Minnelied ( manigerleie blute ꝛc. ) von Winli 2. 22. hat Stollen von 10 , Abgeſ . von 8 , alſo auch 28 R , in ihm iſt der Refrain offenbar eine vom Bau des Abgefangs unabhaͤngige Zuthat ."
        with open("../../conf/dataset/spelling_normalization/german.yaml", mode="r", encoding="utf8") as f:
            spelling_normalization = yaml.safe_load(f)
            print(spelling_normalization)
        preprocessor = Toklem(spelling_normalization=spelling_normalization)
        lemma = "Abgesang"
        new_context, start, end = preprocessor.preprocess(
            context=context, lemma=lemma, index=35
        )
        slice = new_context[start:end+1]
        self.assertEqual(slice, lemma)
        self.assertEqual(start, 164)
        self.assertEqual(end, 171)

    def test_toklem_preprocessing_shorter(self) -> None:
        context = "He agrees that it could 've won"
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Toklem(spelling_normalization=translation_table)
        lemma = "agree"
        new_context, start, end = preprocessor.preprocess(
            context=context, lemma=lemma, index=1
        )

        slice = new_context[start:end+1]
        self.assertEqual(new_context, "He agree that it could've won")
        self.assertEqual(start, 3)
        self.assertEqual(end, 7)
        self.assertEqual(slice, lemma)

'''
class TestSpellingNormalizationLemmatize(unittest.TestCase):
    """12 unit tests for each function in Lemmatize in spelling normalization.
    """    
    def test_character_indices_recalculation_one_after(self) -> None:
        context = "I do n't agree that it could 've won"
        target = "agree"
        span = next(re.finditer(pattern=target, string=context))
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Lemmatize(spelling_normalization=translation_table)
        _, start = preprocessor.normalize_spelling(
                                context, span.start()
                                )
        self.assertEqual(start, span.start() - 1)
    
    def test_character_indices_recalculation_several_after(self) -> None:
        context = "I do n't, I ca n't, I would n't agree that it could 've won"
        target = "agree"
        span = next(re.finditer(pattern=target, string=context))
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Lemmatize(spelling_normalization=translation_table)
        _, start = preprocessor.normalize_spelling(context, span.start())
        self.assertEqual(start, span.start() - 3)


    def test_character_indices_recalculation_several_around(self) -> None:
        context = "I do n't, I can, I would n't agree that it could 've won"
        target = "can"
        span = next(re.finditer(pattern=target, string=context))
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Lemmatize(spelling_normalization=translation_table)
        _, start = preprocessor.normalize_spelling(context, span.start())
        self.assertEqual(start, span.start() - 1)


    def test_character_indices_recalculation_before(self) -> None:
        context = "I agree do n't that it could 've won"
        target = "agree"
        span = next(re.finditer(pattern=target, string=context))
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Lemmatize(spelling_normalization=translation_table)
        _, start = preprocessor.normalize_spelling(context, span.start())
        self.assertEqual(start, span.start())

    def test_character_indices_recalculation_introduce_blanks(self) -> None:
        context = "I agree don't that it could've won"
        target = "that"
        span = next(re.finditer(pattern=target, string=context))
        translation_table = {"n't": " n't", "'ve": " 've"}
        preprocessor = Lemmatize(spelling_normalization=translation_table)
        _, start = preprocessor.normalize_spelling(context, span.start())
        self.assertEqual(start, span.start() + 1)


    def test_lemmatize_preprocessing_longer(self) -> None:
        context = "He agrees that it could 've won"
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Lemmatize(spelling_normalization=translation_table)
        # lemma = "agreess"
        new_context, start, end = preprocessor.preprocess(
            context=context, index=1
        )
        # TypeError: Lemmatize.preprocess() got an unexpected keyword argument 'lemma'
        # slice = new_context[start:end+1]
        self.assertEqual(new_context, "He agreess that it could've won")
        # AssertionError: "He agrees that it could've won" != "He agreess that it could've won"
        self.assertEqual(start, 3)
        self.assertEqual(end, 9)
        # self.assertEqual(slice, lemma)

    def test_lemmatize_preprocessing_equal(self) -> None:
        context = "He agrees that it could 've won"
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Lemmatize(spelling_normalization=translation_table)
        # lemma = "agrees"
        new_context, start, end = preprocessor.preprocess(
            context=context, index=1
        )
        # TypeError: Lemmatize.preprocess() got an unexpected keyword argument 'lemma'
        # slice = new_context[start:end+1]
        self.assertEqual(new_context, "He agrees that it could've won")
        # self.assertEqual(slice, lemma)
        self.assertEqual(start, 3)
        self.assertEqual(end, 8)

    def test_toklem_preprocessing_shorter_no_sp_norm(self) -> None:
        context = "He agrees that it could 've won"
        preprocessor = Lemmatize(spelling_normalization=None)
        # lemma = "agree"
        new_context, start, end = preprocessor.preprocess(
            context=context, index=1
        )
        # slice = new_context[start:end+1]
        self.assertEqual(new_context, "He agree that it could 've won")
        # AssertionError: "He agrees that it could 've won" != "He agree that it could 've won"
        self.assertEqual(start, 3)
        self.assertEqual(end, 7)
        # self.assertEqual(slice, lemma)


    def test_real_case(self) -> None:
        context = "In the afternoons , tea and what appeared to be crumpets were set out near a patio ."
        preprocessor = Lemmatize(spelling_normalization=None)
        # lemma = "afternoon"
        new_context, start, end = preprocessor.preprocess(
            context=context, index=2
        )
        # TypeError: Lemmatize.preprocess() got an unexpected keyword argument 'lemma'
        # slice = new_context[start:end+1]
        self.assertEqual(new_context, "In the afternoon , tea and what appeared to be crumpets were set out near a patio .")
        # AssertionError: 'In the afternoons , tea and what appeared to be crumpets [23 chars]io .' != 'In the afternoon , tea and what appeared to be crumpets w[22 chars]io .'
        self.assertEqual(start, 7)
        self.assertEqual(end, 15)
        # self.assertEqual(slice, lemma)
        

    def test_real_case_german(self) -> None:
        context = "Aufgeſang fuͤr Stoll , welches Docen durch ein Beiſpiel von 1515 rechtfertigt ( Note 11. ſeiner Abh. ) waͤre eine ſehr paſſende Benennung und entſpraͤche dem Abgeſang ."
        with open("../../conf/dataset/spelling_normalization/german.yaml", mode="r", encoding="utf8") as f:
            spelling_normalization = yaml.safe_load(f)
        preprocessor = Lemmatize(spelling_normalization=spelling_normalization)
        # lemma = "Abgesang"
        new_context, start, end = preprocessor.preprocess(
            context=context, index=26
        )
        # TypeError: Lemmatize.preprocess() got an unexpected keyword argument 'lemma'
        slice = new_context[start:end+1]
        # self.assertEqual(slice, lemma)
        self.assertEqual(start, 155)
        self.assertEqual(end, 162)

    def test_real_case_german_2(self) -> None:
        context = "Seltner werden die Lieder , wo es hoͤher ſteigt , doch kenne ich zwei mit ſiebenreimigen Stollen , naͤmlich eins des Nifen 1. 23 . ( ſeht an die heide ꝛc. ) , wo der Abgeſang 9 , der ganze Ton alſo 23 Reime zaͤhlt , und eines von Canzler 2. 244. ( helfent mir ꝛc. ) , wo im Abgeſang 14 , im Ganzen 28 Reime ſtecken . Das Minnelied ( manigerleie blute ꝛc. ) von Winli 2. 22. hat Stollen von 10 , Abgeſ . von 8 , alſo auch 28 R , in ihm iſt der Refrain offenbar eine vom Bau des Abgefangs unabhaͤngige Zuthat ."
        with open("../../conf/dataset/spelling_normalization/german.yaml", mode="r", encoding="utf8") as f:
            spelling_normalization = yaml.safe_load(f)
            print(spelling_normalization)
        preprocessor = Lemmatize(spelling_normalization=spelling_normalization)
        # lemma = "Abgesang"
        new_context, start, end = preprocessor.preprocess(
            context=context, index=35
        )
        # TypeError: Lemmatize.preprocess() got an unexpected keyword argument 'lemma'
        slice = new_context[start:end+1]
        # self.assertEqual(slice, lemma)
        self.assertEqual(start, 164)
        self.assertEqual(end, 171)

    def test_lemmatize_preprocessing_shorter(self) -> None:
        context = "He agrees that it could 've won"
        translation_table = {" n't": "n't", " 've": "'ve"}
        preprocessor = Lemmatize(spelling_normalization=translation_table)
        # lemma = "agree"
        new_context, start, end = preprocessor.preprocess(
            context=context, index=1
        )
        # TypeError: Lemmatize.preprocess() got an unexpected keyword argument 'lemma'
        # slice = new_context[start:end+1]
        self.assertEqual(new_context, "He agree that it could've won")
        # AssertionError: "He agrees that it could've won" != "He agree that it could've won"
        self.assertEqual(start, 3)
        self.assertEqual(end, 7)
        # self.assertEqual(slice, lemma)
'''

if __name__ == '__main__':
    unittest.main()
