from src.preprocessing import Lemmatize, Toklem
import yaml
import re


def test_character_indices_recalculation_one_after() -> None:
    context = "I do n't agree that it could 've won"
    target = "agree"
    span = next(re.finditer(pattern=target, string=context))
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    new_context, start = preprocessor.normalize_spelling(
        context, span.start()
    )
    assert start == span.start() - 1


def test_character_indices_recalculation_several_after() -> None:
    context = "I do n't, I ca n't, I would n't agree that it could 've won"
    target = "agree"
    span = next(re.finditer(pattern=target, string=context))
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    _, start = preprocessor.normalize_spelling(context, span.start())
    assert start == span.start() - 3


def test_character_indices_recalculation_several_around() -> None:
    context = "I do n't, I can, I would n't agree that it could 've won"
    target = "can"
    span = next(re.finditer(pattern=target, string=context))
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    _, start = preprocessor.normalize_spelling(context, span.start())
    assert start == span.start() - 1


def test_character_indices_recalculation_before() -> None:
    context = "I agree do n't that it could 've won"
    target = "agree"
    span = next(re.finditer(pattern=target, string=context))
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    _, start = preprocessor.normalize_spelling(context, span.start())
    assert start == span.start()

def test_character_indices_recalculation_introduce_blanks() -> None:
    context = "I agree don't that it could've won"
    target = "that"
    span = next(re.finditer(pattern=target, string=context))
    translation_table = {"n't": " n't", "'ve": " 've"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    _, start = preprocessor.normalize_spelling(context, span.start())
    assert start == span.start() + 1


def test_toklem_preprocessing_longer() -> None:
    context = "He agrees that it could 've won"
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    lemma = "agreess"
    new_context, start, end = preprocessor.preprocess(
        context=context, lemma=lemma, index=1
    )
    slice = new_context[start:end+1]
    assert new_context == "He agreess that it could've won"
    assert start == 3
    assert end == 9
    assert slice == lemma

def test_toklem_preprocessing_equal() -> None:
    context = "He agrees that it could 've won"
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    lemma = "agrees"
    new_context, start, end = preprocessor.preprocess(
        context=context, lemma=lemma, index=1
    )
    slice = new_context[start:end+1]
    assert new_context == "He agrees that it could've won"
    assert slice == lemma
    assert start == 3
    assert end == 8

def test_toklem_preprocessing_shorter_no_sp_norm() -> None:
    context = "He agrees that it could 've won"
    preprocessor = Toklem(spelling_normalization=None)
    lemma = "agree"

    new_context, start, end = preprocessor.preprocess(
        context=context, lemma=lemma, index=1
    )
    slice = new_context[start:end+1]
    assert new_context == "He agree that it could 've won"
    assert start == 3
    assert end == 7
    assert slice == lemma


def test_real_case() -> None:
    context = "In the afternoons , tea and what appeared to be crumpets were set out near a patio ."
    preprocessor = Toklem(spelling_normalization=None)
    lemma = "afternoon"
    new_context, start, end = preprocessor.preprocess(
        context=context, lemma=lemma, index=2
    )
    slice = new_context[start:end+1]
    assert new_context == "In the afternoon , tea and what appeared to be crumpets were set out near a patio ."
    assert start == 7
    assert end == 15
    assert slice == lemma
    

def test_real_case_german() -> None:
    context = "Aufgeſang fuͤr Stoll , welches Docen durch ein Beiſpiel von 1515 rechtfertigt ( Note 11. ſeiner Abh. ) waͤre eine ſehr paſſende Benennung und entſpraͤche dem Abgeſang ."
    with open("conf/dataset/spelling_normalization/german.yaml", mode="r", encoding="utf8") as f:
        spelling_normalization = yaml.safe_load(f)
    preprocessor = Toklem(spelling_normalization=spelling_normalization)
    lemma = "Abgesang"
    new_context, start, end = preprocessor.preprocess(
        context=context, lemma=lemma, index=26
    )
    slice = new_context[start:end+1]
    assert slice == lemma
    assert start == 155
    assert end == 162

def test_real_case_german_2() -> None:
    context = "Seltner werden die Lieder , wo es hoͤher ſteigt , doch kenne ich zwei mit ſiebenreimigen Stollen , naͤmlich eins des Nifen 1. 23 . ( ſeht an die heide ꝛc. ) , wo der Abgeſang 9 , der ganze Ton alſo 23 Reime zaͤhlt , und eines von Canzler 2. 244. ( helfent mir ꝛc. ) , wo im Abgeſang 14 , im Ganzen 28 Reime ſtecken . Das Minnelied ( manigerleie blute ꝛc. ) von Winli 2. 22. hat Stollen von 10 , Abgeſ . von 8 , alſo auch 28 R , in ihm iſt der Refrain offenbar eine vom Bau des Abgefangs unabhaͤngige Zuthat ."
    with open("conf/dataset/spelling_normalization/german.yaml", mode="r", encoding="utf8") as f:
        spelling_normalization = yaml.safe_load(f)
        print(spelling_normalization)
    preprocessor = Toklem(spelling_normalization=spelling_normalization)
    lemma = "Abgesang"
    new_context, start, end = preprocessor.preprocess(
        context=context, lemma=lemma, index=35
    )
    slice = new_context[start:end+1]
    assert slice == lemma
    assert start == 164
    assert end == 171




def test_toklem_preprocessing_shorter() -> None:
    context = "He agrees that it could 've won"
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    lemma = "agree"
    new_context, start, end = preprocessor.preprocess(
        context=context, lemma=lemma, index=1
    )

    slice = new_context[start:end+1]
    assert new_context == "He agree that it could've won"
    assert start == 3
    assert end == 7
    assert slice == lemma
