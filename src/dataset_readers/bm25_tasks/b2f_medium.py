import re
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import json, os
from src.utils.app import App
from nltk.tokenize import word_tokenize
from src.utils.dataset_utils import load_train_dataset

field_getter = App()


@field_getter.add("q")
def get_question(entry):

    return b2f_mediumBM25Task.norm(entry['question'])


@field_getter.add("qa")
def get_qa(entry):
    return b2f_mediumBM25Task.norm(f"{entry['question']} {entry['target']}")


@field_getter.add("a")
def get_decomp(entry):
    return b2f_mediumBM25Task.norm(entry['target'])


class b2f_mediumBM25Task:
    name = 'b2f_medium'

    def __init__(self, dataset_split, setup_type, ds_size=None):
        self.setup_type = setup_type
        self.get_field = field_getter.functions[self.setup_type]
        self.dataset_split = dataset_split
        dataset = load_dataset("mengmengmmm/B2F_medium")
        print(dataset)
        self.train_dataset = load_train_dataset(dataset, size=ds_size)
        if self.dataset_split == "train":
            self.dataset = self.train_dataset
        else:
            self.dataset = list(dataset[self.dataset_split])
        self.corpus = None

    def get_corpus(self):
        if self.corpus is None:
            self.corpus = [self.get_field(entry) for entry in self.train_dataset]
        return self.corpus

    @classmethod
    def norm(cls, text):

        return word_tokenize(text)
