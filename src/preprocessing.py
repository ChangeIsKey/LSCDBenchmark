from pandas import Series

def keep_intact(s: Series, _: dict[str, str]) -> tuple[str, int, int]:
    start, end = tuple(map(int, s.indexes_target_token.split(":")))
    return s.context, start, end

def clean_context(context: str, translation_table: dict[str, str]) -> str:
    for key, replacement in translation_table.items():
        context = context.replace(key, replacement)
    return context

def char_indices(token_idx: int, tokens: list[str], target: str) -> tuple[int, int]:
    char_idx = -1
    for i, token in enumerate(tokens):
        if i == token_idx:
            # char_idx will be one index to the left of the target, so we need to add 1
            start = char_idx + 1
            end = start + len(target)
            return start, end
        else:
            char_idx += len(token) + 1  # plus one space

    raise ValueError


def toklem(s: Series, translation_table: dict[str, str]) -> tuple[str, int, int]:
    tokens = clean_context(s.context_tokenized, translation_table).split()
    target = s.lemma.split("_")[0]
    start, end = char_indices(
        token_idx=s.indexes_target_token_tokenized, tokens=tokens, target=target
    )
    tokens[int(s.indexes_target_token_tokenized)] = target
    return " ".join(tokens), start, end


def lemmatize(s: Series, translation_table: dict[str, str]) -> tuple[str, int, int]:
    context_preprocessed = clean_context(s.context_lemmatized, translation_table)
    tokens = context_preprocessed.split()

    # the spanish dataset has an index column for the lemmatized contexts, but all the others don't
    idx = s.get(
        "indexes_target_token_lemmatized", default=s.indexes_target_token_tokenized
    )
    start, end = char_indices(token_idx=idx, tokens=tokens, target=tokens[idx])
    return context_preprocessed, start, end


def tokenize(s: Series, translation_table: dict[str, str]) -> tuple[str, int, int]:
    tokens = clean_context(s.context_tokenized, translation_table).split()
    idx = s.indexes_target_token_tokenized
    start, end = char_indices(token_idx=idx, tokens=tokens, target=tokens[idx])
    return " ".join(tokens), start, end
