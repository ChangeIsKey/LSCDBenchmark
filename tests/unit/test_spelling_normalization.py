from src.preprocessing import Lemmatize, Toklem
import re


def test_character_indices_recalculation_one_after() -> None:
    context = "I do n't agree that it could 've won"
    target = "agree"
    span = next(re.finditer(pattern=target, string=context))
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    new_context, start, end = preprocessor.normalize_spelling(
        context, span.start(), span.end()
    )
    assert start == span.start() - 1
    assert end == span.end() - 1


def test_character_indices_recalculation_several_after() -> None:
    context = "I do n't, I ca n't, I would n't agree that it could 've won"
    target = "agree"
    span = next(re.finditer(pattern=target, string=context))
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    _, start, end = preprocessor.normalize_spelling(context, span.start(), span.end())
    assert start == span.start() - 3
    assert end == span.end() - 3


def test_character_indices_recalculation_several_around() -> None:
    context = "I do n't, I can, I would n't agree that it could 've won"
    target = "can"
    span = next(re.finditer(pattern=target, string=context))
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    _, start, end = preprocessor.normalize_spelling(context, span.start(), span.end())
    assert start == span.start() - 1
    assert end == span.end() - 1


def test_character_indices_recalculation_before() -> None:
    context = "I agree do n't that it could 've won"
    target = "agree"
    span = next(re.finditer(pattern=target, string=context))
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    _, start, end = preprocessor.normalize_spelling(context, span.start(), span.end())
    assert start == span.start()
    assert end == span.end()

def test_character_indices_recalculation_introduce_blanks() -> None:
    context = "I agree don't that it could've won"
    target = "that"
    span = next(re.finditer(pattern=target, string=context))
    translation_table = {"n't": " n't", "'ve": " 've"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    _, start, end = preprocessor.normalize_spelling(context, span.start(), span.end())
    assert start == span.start() + 1
    assert end == span.end() + 1


def test_toklem_preprocessing_longer() -> None:
    context = "He agrees that it could 've won"
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    new_context, start, end = preprocessor.preprocess(
        context=context, lemma="agreess", index=1
    )
    assert new_context == "He agreess that it could've won"
    assert start == 3
    assert end == 9



def test_toklem_preprocessing_shorter() -> None:
    context = "He agrees that it could 've won"
    translation_table = {" n't": "n't", " 've": "'ve"}
    preprocessor = Toklem(spelling_normalization=translation_table)
    new_context, start, end = preprocessor.preprocess(
        context=context, lemma="agree", index=1
    )

    assert new_context == "He agree that it could've won"
    assert start == 3
    assert end == 7
