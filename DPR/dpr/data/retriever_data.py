import collections
import csv
import json
import logging
import os.path
import pickle
from typing import Dict


import hydra
import jsonlines
import torch
from omegaconf import DictConfig
from dpr.utils.data_utils import App
from datasets import load_dataset, load_from_disk
import datasets

from dpr.utils.data_utils import load_train_dataset, get_one_prompt

from dpr.data.biencoder_data import (
    BiEncoderPassage,
    normalize_passage,
    normalize_question,
    get_dpr_files,
    read_nq_tables_jsonl,
    split_tables_to_chunks,
)

logger = logging.getLogger(__name__)
QASample = collections.namedtuple("QuerySample", ["query", "id", "answers"])
TableChunk = collections.namedtuple("TableChunk", ["text", "title", "table_id"])



class RetrieverData(torch.utils.data.Dataset):
    def __init__(self, file: str):
        """
        :param file: - real file name or the resource name as they are defined in download_data.py
        """
        self.file = file
        self.data_files = []

    def load_data(self):
        self.data_files = get_dpr_files(self.file)
        assert (
            len(self.data_files) == 1
        ), "RetrieverData source currently works with single files only. Files specified: {}".format(
            self.data_files
        )
        self.file = self.data_files[0]


class QASrc(RetrieverData):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file)
        self.data = None
        self.selector = hydra.utils.instantiate(selector) if selector else None
        self.special_query_token = special_query_token
        self.query_special_suffix = query_special_suffix

    def __getitem__(self, index) -> QASample:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _process_question(self, question: str):
        # as of now, always normalize query
        question = normalize_question(question)
        if self.query_special_suffix and not question.endswith(
            self.query_special_suffix
        ):
            question += self.query_special_suffix
        return question


class CsvQASrc(QASrc):
    def __init__(
        self,
        file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_col = question_col
        self.answers_col = answers_col
        self.id_col = id_col

    def load_data(self):
        super().load_data()
        data = []
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                question = row[self.question_col]
                answers = eval(row[self.answers_col])
                id = None
                if self.id_col >= 0:
                    id = row[self.id_col]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data




class JsonlQASrc(QASrc):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        question_attr: str = "question",
        answers_attr: str = "answers",
        id_attr: str = "id",
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_attr = question_attr
        self.answers_attr = answers_attr
        self.id_attr = id_attr

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                answers = jline[self.answers_attr] if self.answers_attr in jline else []
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class KiltCsvQASrc(CsvQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(
            file,
            question_col,
            answers_col,
            id_col,
            selector,
            special_query_token,
            query_special_suffix,
        )
        self.kilt_gold_file = kilt_gold_file


class KiltJsonlQASrc(JsonlQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_attr: str = "input",
        answers_attr: str = "answer",
        id_attr: str = "id",
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(
            file,
            selector,
            question_attr,
            answers_attr,
            id_attr,
            special_query_token,
            query_special_suffix,
        )
        self.kilt_gold_file = kilt_gold_file

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                out = jline["output"]
                answers = [o["answer"] for o in out if "answer" in o]
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class TTS_ASR_QASrc(QASrc):
    def __init__(self, file: str, trans_file: str):
        super().__init__(file)
        self.trans_file = trans_file

    def load_data(self):
        super().load_data()
        orig_data_dict = {}
        with open(self.file, "r") as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            id = 0
            for row in reader:
                question = row[0]
                answers = eval(row[1])
                orig_data_dict[id] = (question, answers)
                id += 1
        data = []
        with open(self.trans_file, "r") as tfile:
            reader = csv.reader(tfile, delimiter="\t")
            for r in reader:
                row_str = r[0]
                idx = row_str.index("(None-")
                q_id = int(row_str[idx + len("(None-") : -1])
                orig_data = orig_data_dict[q_id]
                answers = orig_data[1]
                q = row_str[:idx].strip().lower()
                data.append(QASample(q, idx, answers))
        self.data = data


class CsvCtxSrc(RetrieverData):
    def __init__(
        self,
        file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file)
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        super().load_data()
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                if row[self.id_col] == "id":
                    continue
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[self.id_col])
                else:
                    sample_id = row[self.id_col]
                passage = row[self.text_col]
                if self.normalize:
                    passage = normalize_passage(passage)
                ctxs[sample_id] = BiEncoderPassage(passage, row[self.title_col])

dataset_dict = App()






@dataset_dict.add("concode")
def get_concode():
    dataset = load_dataset("mengmengmmm/concode_trainuse")
    return dataset

@dataset_dict.add("java2cs")
def get_java2cs():
    dataset = load_dataset("mengmengmmm/java2cs_trainuse")
    return dataset
    
@dataset_dict.add("csn_ruby")
def get_csn_ruby():
    dataset = load_dataset("mengmengmmm/csn_ruby_trainuse")
    return dataset

@dataset_dict.add("csn_python")
def get_csn_python():
    dataset = load_dataset("mengmengmmm/csn_python_trainuse")
    return dataset


@dataset_dict.add("csn_python_slice1")
def get_csn_python_slice1():
    dataset = load_dataset("mengmengmmm/csn_python_trainuse_slice1")
    return dataset

@dataset_dict.add("csn_php")
def get_csn_php():
    dataset = load_dataset("mengmengmmm/csn_php_trainuse")
    return dataset

@dataset_dict.add("csn_js")
def get_csn_js():
    dataset = load_dataset("mengmengmmm/csn_js_trainuse")
    return dataset    
    
@dataset_dict.add("csn_go")
def get_csn_go():
    dataset = load_dataset("mengmengmmm/csn_go_trainuse")
    return dataset    

@dataset_dict.add("csn_go_slice1")
def get_csn_go_slice1():
    dataset = load_dataset("mengmengmmm/csn_go_trainuse_slice1")
    return dataset

@dataset_dict.add("csn_go_slice2")
def get_csn_go_slice2():
    dataset = load_dataset("mengmengmmm/csn_go_trainuse_slice2")
    return dataset


@dataset_dict.add("tlc")
def get_tlc():
    dataset = load_dataset("mengmengmmm/tlc")
    return dataset    

@dataset_dict.add("tlc_slice1")
def get_tlc_slice1():
    dataset = load_dataset("mengmengmmm/tlc_slice1")
    return dataset

@dataset_dict.add("tlc_slice2")
def get_tlc_slice2():
    dataset = load_dataset("mengmengmmm/tlc_slice2")
    return dataset

@dataset_dict.add("csn_java")
def get_csn_java():
    dataset = load_dataset("mengmengmmm/csn_java")
    return dataset   

@dataset_dict.add("csn_java_slice1")
def get_csn_java_slice1():
    dataset = load_dataset("mengmengmmm/csn_java_slice1")
    return dataset


@dataset_dict.add("csn_java_slice2")
def get_csn_java_slice2():
    dataset = load_dataset("mengmengmmm/csn_java_slice2")
    return dataset
    
    
@dataset_dict.add("csn_java_slice3")
def get_csn_java_slice3():
    dataset = load_dataset("mengmengmmm/csn_java_slice3")
    return dataset
    
    
@dataset_dict.add("csn_java_slice4")
def get_csn_java_slice4():
    dataset = load_dataset("mengmengmmm/csn_java_slice4")
    return dataset


@dataset_dict.add("conala")
def get_python():
    dataset = load_dataset("mengmengmmm/conala")
    return dataset    

@dataset_dict.add("b2f_medium")
def get_python():
    dataset = load_dataset("mengmengmmm/B2F_medium")
    return dataset   

@dataset_dict.add("b2f_small")
def get_python():
    dataset = load_dataset("mengmengmmm/B2F_small")
    return dataset  









fields_dict = {

    "csn_python": {"question_attr": "question", "answers_attr": "target"},
    "csn_python_slice1": {"question_attr": "question", "answers_attr": "target"},
    "csn_go_slice1": {"question_attr": "question", "answers_attr": "target"},
    "csn_go_slice2": {"question_attr": "question", "answers_attr": "target"},
    "csn_go": {"question_attr": "question", "answers_attr": "target"},
    "csn_js": {"question_attr": "question", "answers_attr": "target"},
    "csn_php": {"question_attr": "question", "answers_attr": "target"},
    "csn_java": {"question_attr": "question", "answers_attr": "target"},
    "csn_java_slice1": {"question_attr": "question", "answers_attr": "target"},
    "csn_java_slice2": {"question_attr": "question", "answers_attr": "target"},
    "csn_java_slice3": {"question_attr": "question", "answers_attr": "target"},
    "csn_java_slice4": {"question_attr": "question", "answers_attr": "target"},
    "csn_ruby": {"question_attr": "question", "answers_attr": "target"},
    "tlc": {"question_attr": "question", "answers_attr": "target"},
    "tlc_slice1": {"question_attr": "question", "answers_attr": "target"},
    "tlc_slice2": {"question_attr": "question", "answers_attr": "target"},
    "b2f_small": {"question_attr": "question", "answers_attr": "target"},
    "b2f_medium": {"question_attr": "question", "answers_attr": "target"},
    "conala": {"question_attr": "question", "answers_attr": "target"},



}




class EPRQASrc(QASrc):
    def __init__(
        self,
        dataset_split,
        task_name,
        ds_size=None,
        file =  "",
        selector: DictConfig = None,
        question_attr: str = "question",
        answers_attr: str = "answers",
        id_attr: str = "id",
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.task_name = task_name
        
        self.dataset_split = dataset_split
        # assert  self.dataset_split in ["train","validation","test","debug","test_asset","test_turk","test_wiki"]
        self.dataset = dataset_dict.functions[self.task_name]()
        if self.dataset_split=="train":
            self.data = load_train_dataset(self.dataset,size=ds_size)
        else:
            self.data = list(self.dataset[self.dataset_split])
        if ds_size is not None:
            assert len(self.data) == ds_size 
        
        self.question_attr = fields_dict[self.task_name]["question_attr"]
        self.answers_attr = fields_dict[self.task_name]["answers_attr"]
        self.id_attr = id_attr

    def load_data(self):
        # super().load_data()
        data = []
        # with jsonlines.open(self.file, mode="r") as jsonl_reader:
        for id, jline in enumerate(self.data):
            question = jline[self.question_attr]
            answers = [str(jline[self.answers_attr])]
            # id = None
            # if self.id_attr in jline:
                # id = jline[self.id_attr]
            data.append(QASample(self._process_question(question), id, answers))
        self.data = data




def reformat(text):
    return " ".join([f"{i+1}#) {x.strip()}" for i,x in enumerate(text.split(";"))])

app = App()










@app.add("b2f_small_q")
def get_b2f_small_question(entry):
    return "Fixed on the buggy. Buggy: " + entry['question']

@app.add("b2f_small_qa")
def get_b2f_small_question_decomp(entry):
    return f"Fixed on the buggy. Buggy: {entry['question']}\tFixed: {entry['target']}"

@app.add("b2f_small_a")
def get_b2f_small_decomp(entry):
    return entry['target']


@app.add("b2f_medium_q")
def get_b2f_medium_question(entry):
    return "Fixed on the buggy. Buggy: " + entry['question']

@app.add("b2f_medium_qa")
def get_b2f_medium_question_decomp(entry):
    return f"Fixed on the buggy. Buggy: {entry['question']}\tFixed: {entry['target']}"

@app.add("b2f_medium_a")
def get_b2f_medium_decomp(entry):
    return entry['target']

@app.add("conala_q")
def get_conala_question(entry):
    return "Code on the comment. Comment: " + entry['question']

@app.add("conala_qa")
def get_conala_question_decomp(entry):
    return f"Code on the comment. Comment: {entry['question']}\tCode: {entry['target']}"

@app.add("conala_a")
def get_conala_decomp(entry):
    return entry['target']


@app.add("tlc_q")
def get_tlc_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("tlc_qa")
def get_tlc_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("tlc_a")
def get_tlc_decomp(entry):
    return entry['target']


@app.add("tlc_slice1_q")
def get_tlc_slice1_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("tlc_slice1_qa")
def get_tlc_slice1_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("tlc_slice1_a")
def get_tlc_slice1_decomp(entry):
    return entry['target']


@app.add("tlc_slice2_q")
def get_tlc_slice2_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("tlc_slice2_qa")
def get_tlc_slice2_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("tlc_slice2_a")
def get_tlc_slice2_decomp(entry):
    return entry['target']

@app.add("csn_python_slice1_q")
def get_csn_python_slice1_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_python_slice1_qa")
def get_csn_python_slice1_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_python_slice1_a")
def get_csn_python_slice1_decomp(entry):
    return entry['target']


@app.add("csn_python_q")
def get_csn_python_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_python_qa")
def get_csn_python_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_python_a")
def get_csn_python_decomp(entry):
    return entry['target']


@app.add("csn_go_slice1_q")
def get_csn_go_slice1_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_go_slice1_qa")
def get_csn_go_slice1_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_go_slice1_a")
def get_csn_go_slice1_decomp(entry):
    return entry['target']

@app.add("csn_go_slice2_q")
def get_csn_go_slice2_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_go_slice2_qa")
def get_csn_go_slice2_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_go_slice2_a")
def get_csn_go_slice2_decomp(entry):
    return entry['target']


@app.add("csn_java_q")
def get_csn_java_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_java_qa")
def get_csn_java_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_java_a")
def get_csn_java_decomp(entry):
    return entry['target']


@app.add("csn_java_slice1_q")
def get_csn_java_slice1_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_java_slice1_qa")
def get_csn_java_slice1_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_java_slice1_a")
def get_csn_java_slice1_decomp(entry):
    return entry['target']




@app.add("csn_java_slice2_q")
def get_csn_java_slice2_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_java_slice2_qa")
def get_csn_java_slice2_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_java_slice2_a")
def get_csn_java_slice2_decomp(entry):
    return entry['target']


@app.add("csn_java_slice3_q")
def get_csn_java_slice3_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_java_slice3_qa")
def get_csn_java_slice3_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_java_slice3_a")
def get_csn_java_slice3_decomp(entry):
    return entry['target']



@app.add("csn_java_slice4_q")
def get_csn_java_slice4_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_java_slice4_qa")
def get_csn_java_slice4_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_java_slice4_a")
def get_csn_java_slice4_decomp(entry):
    return entry['target']


@app.add("csn_ruby_q")
def get_csn_ruby_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_ruby_qa")
def get_csn_ruby_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_ruby_a")
def get_csn_ruby_decomp(entry):
    return entry['target']


@app.add("csn_js_q")
def get_csn_js_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_js_qa")
def get_csn_js_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_js_a")
def get_csn_js_decomp(entry):
    return entry['target']


@app.add("csn_php_q")
def get_csn_php_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_php_qa")
def get_csn_php_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_php_a")
def get_csn_php_decomp(entry):
    return entry['target']


@app.add("csn_go_q")
def get_csn_go_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_go_qa")
def get_csn_go_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_go_a")
def get_csn_go_decomp(entry):
    return entry['target']









class EPRCtxSrc(RetrieverData):
    def __init__(
        self,
        task_name,
        setup_type,
        ds_size=None,
        file = "",
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file)
        self.setup_type = setup_type
        assert self.setup_type in ["q","qa","a"]
        self.file = file
        self.task_name = task_name
        self.dataset = dataset_dict.functions[self.task_name]()
        self.get_field = app.functions[f"{self.task_name}_{self.setup_type}"]
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.data = load_train_dataset(self.dataset,size=ds_size)
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        # with open(self.file) as ifile:
        #     reader = json.load(ifile)
        for sample_id,entry in enumerate(self.data):
            passage = self.get_field(entry)
            if self.normalize:
                passage = normalize_passage(passage)
            ctxs[sample_id] = BiEncoderPassage(passage, "")

class JsonCtxSrc(RetrieverData):
    def __init__(
        self,
        file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file)
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        super().load_data()
        with open(self.file) as ifile:
            reader = json.load(ifile)
            for row in reader:
                sample_id = row["id"]
                passage = row['text']
                if self.normalize:
                    passage = normalize_passage(passage)
                ctxs[sample_id] = BiEncoderPassage(passage, row['title'])


class KiltCsvCtxSrc(CsvCtxSrc):
    def __init__(
        self,
        file: str,
        mapping_file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(
            file, id_col, text_col, title_col, id_prefix, normalize=normalize
        )
        self.mapping_file = mapping_file

    def convert_to_kilt(self, kilt_gold_file, dpr_output, kilt_out_file):
        logger.info("Converting to KILT format file: %s", dpr_output)

        with open(dpr_output, "rt") as fin:
            dpr_output = json.load(fin)

        with jsonlines.open(kilt_gold_file, "r") as reader:
            kilt_gold_file = list(reader)
        assert len(kilt_gold_file) == len(dpr_output)
        map_path = self.mapping_file
        with open(map_path, "rb") as fin:
            mapping = pickle.load(fin)

        with jsonlines.open(kilt_out_file, mode="w") as writer:
            for dpr_entry, kilt_gold_entry in zip(dpr_output, kilt_gold_file):
                assert dpr_entry["question"] == kilt_gold_entry["input"]
                provenance = []
                for ctx in dpr_entry["ctxs"]:
                    wikipedia_id, end_paragraph_id = mapping[int(ctx["id"])]
                    provenance.append(
                        {
                            "wikipedia_id": wikipedia_id,
                            "end_paragraph_id": end_paragraph_id,
                        }
                    )
                kilt_entry = {
                    "id": kilt_gold_entry["id"],
                    "input": dpr_entry["question"],
                    "output": [{"provenance": provenance}],
                }
                writer.write(kilt_entry)

        logger.info("Saved KILT formatted results to: %s", kilt_out_file)


class JsonlTablesCtxSrc(object):
    def __init__(
        self,
        file: str,
        tables_chunk_sz: int = 100,
        split_type: str = "type1",
        id_prefix: str = None,
    ):
        self.tables_chunk_sz = tables_chunk_sz
        self.split_type = split_type
        self.file = file
        self.id_prefix = id_prefix

    def load_data_to(self, ctxs: Dict):
        docs = {}
        logger.info("Parsing Tables data from: %s", self.file)
        tables_dict = read_nq_tables_jsonl(self.file)
        table_chunks = split_tables_to_chunks(
            tables_dict, self.tables_chunk_sz, split_type=self.split_type
        )
        for chunk in table_chunks:
            sample_id = self.id_prefix + str(chunk[0])
            docs[sample_id] = TableChunk(chunk[1], chunk[2], chunk[3])
        logger.info("Loaded %d tables chunks", len(docs))
        ctxs.update(docs)