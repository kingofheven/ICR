from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import re
import json, os
from src.utils.dataset_utils import load_train_dataset


class java2csScorerTask:
    name = "java2cs"
    question_field = "question"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        dataset = load_dataset("mengmengmmm/java2cs_trainuse")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        self.postfix = "Cs: "

    def get_fields(self, entry, index=-1):
        question_prefix = "Java: "
        answer_prefix = "Cs: "
        test_question = question_prefix + entry['test_question']
        question = question_prefix + entry['question']
        decomposition = answer_prefix + entry['target']
        test_decomposition = entry['test_target']
        return question, decomposition, test_question, test_decomposition
