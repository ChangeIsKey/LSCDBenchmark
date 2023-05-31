import sys
sys.path.insert(0, ".")

from src.preprocessing import Lemmatize, Toklem, Raw, Normalize, ContextPreprocessor
import yaml

import unittest


class TestPreprocessingDe(unittest.TestCase):        

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

if __name__ == '__main__':
    unittest.main()
