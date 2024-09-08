import types
import pathlib
from os.path import dirname, isfile, join
import os
import glob
import json



import os

from datasets import load_from_disk, Dataset, concatenate_datasets
from src.utils.dataset_utils import load_train_dataset

import json
from src.utils.tokenizer_utils import get_length


def set_length(example, idx,**kwargs):
    tokenizer = kwargs['tokenizer']
    question_prefix = "Code: "
    answer_prefix = "Comment: "
    q_field = question_prefix + example['question']
    a_field = answer_prefix + example['target']
    prompt_qa = f"{q_field}\t{a_field}"
    example['prompt_qa'] = prompt_qa
    example['prompt_len'] = get_length(tokenizer,prompt_qa)
    return example



modules = {}
modules_list = glob.glob(join(dirname(__file__), "*.py"))
for path in modules_list:
    if isfile(path) and not path.endswith('__init__.py') and not path.endswith('task_.py'):
        mod_name = pathlib.Path(path).name[:-3]
        module = types.ModuleType(mod_name)
        with open(path) as f:
            module_str = f.read()
        exec(module_str, module.__dict__)
        modules[mod_name] = module

task_list = {}
for module_name, module in modules.items():
    for el in dir(module):
        if el.endswith("InferenceTask"):
            obj = module.__dict__[el]
            task_list[obj.name] = obj


class InferenceTask:
    def __init__(self) -> None:
        pass
    @classmethod
    def from_name(cls,name):
        return task_list[name]


class csn_java_slice1InferenceTask:
    name = "csn_java_slice1"

    def __init__(self, prompt_file, tokenizer, ds_size=None):
        self.prompt_file = prompt_file
        with open(self.prompt_file) as f:
            self.prompts = json.load(f)
        dataset = load_dataset("mengmengmmm/csn_java_slice1")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size, listify=False)
        self.hf_dataset = self.hf_dataset.map(set_length, with_indices=True, fn_kwargs={'tokenizer': tokenizer})
        self.training_dataset = list(self.hf_dataset)
        self.postfix = 'Comment: '

    @classmethod
    def postproccess(cls, string):
        return string

    def get_fields(self, entry):
        question = entry['question']
        answer = entry['target'] if "target" in entry else entry['answers'][0]
        idx_list = [p['id'] for p in entry['ctxs']]
        prompts = self.hf_dataset.select([p['id'] for p in entry['ctxs']])
        return "Code: " + question, answer, prompts['prompt_qa'], prompts['prompt_len'], idx_list