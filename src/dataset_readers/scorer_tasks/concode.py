from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import re
import json, os
from src.utils.dataset_utils import load_train_dataset


class concodeScorerTask:
    name = "concode"
    question_field = "question"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        dataset = load_dataset("mengmengmmm/concode_trainuse")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        self.postfix = "Code: "

    def get_fields(self, entry, index=-1):
        question_prefix = "Comment: "
        answer_prefix = "Code: "
        test_question = question_prefix + entry['test_question']
        question = question_prefix + entry['question']
        decomposition = answer_prefix + entry['target']
        test_decomposition = entry['test_target']
        return question, decomposition, test_question, test_decomposition
