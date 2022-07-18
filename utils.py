from torch import Tensor
from transformers import BertTokenizer, BertModel, logging
from typing import List, Tuple

logging.set_verbosity_error()


def vectorize(contexts: List[str], target_indices: List[Tuple[int, int]], embedding: str, cased: bool) -> Tensor:
    # TODO ask nikolai how to make this more general

    model_name = f'bert-base-{"cased" if cased else "uncased"}'

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    encoded_input = tokenizer(contexts, return_tensors='pt', truncation=True)
    output = model(**encoded_input)
    return output

    # if embedding == "bert":
    #     # if BERT is None:
    #     #     BERT =
    #     model = f"bert-base-{'cased' if cased else 'uncased'}"
    #     vectors = []
    #     for context, indices in zip(contexts, target_indices):
    #         # vectorize
    #         pass
    #
    #     # return vectors
    # else:
    #     pass
