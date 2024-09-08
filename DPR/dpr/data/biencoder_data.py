import collections
import csv
import glob
import logging
import os
import random
from typing import Dict, List, Tuple
from datasets import load_dataset, load_from_disk
import datasets
from dpr.utils.data_utils import load_train_dataset, get_one_prompt


import hydra
import jsonlines
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor as T
import random
from dpr.data.tables import Table
from dpr.utils.data_utils import read_data_from_json_files, Tensorizer,App

logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])


class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]

class BiEncoderSample_for_list_ranking:
    query: str
    sorted_passages: List[BiEncoderPassage]


class RepTokenSelector(object):
    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        raise NotImplementedError


class RepStaticPosTokenSelector(RepTokenSelector):
    def __init__(self, static_position: int = 0):
        self.static_position = static_position

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        return self.static_position


class RepSpecificTokenSelector(RepTokenSelector):
    def __init__(self, token: str = "[CLS]"):
        self.token = token
        self.token_id = None

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        if not self.token_id:
            self.token_id = tenzorizer.get_token_id(self.token)
        token_indexes = (input_ids == self.token_id).nonzero()
        # check if all samples in input_ids has index presence and out a default value otherwise
        bsz = input_ids.size(0)
        if bsz == token_indexes.size(0):
            return token_indexes

        token_indexes_result = []
        found_idx_cnt = 0
        for i in range(bsz):
            if (
                found_idx_cnt < token_indexes.size(0)
                and token_indexes[found_idx_cnt][0] == i
            ):
                # this samples has the special token
                token_indexes_result.append(token_indexes[found_idx_cnt])
                found_idx_cnt += 1
            else:
                logger.warning("missing special token %s", input_ids[i])

                token_indexes_result.append(
                    torch.tensor([i, 0]).to(input_ids.device)
                )  # setting 0-th token, i.e. CLS for BERT as the special one
        token_indexes_result = torch.stack(token_indexes_result, dim=0)
        return token_indexes_result


DEFAULT_SELECTOR = RepStaticPosTokenSelector()


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        selector: DictConfig = None,
        special_token: str = None,
        shuffle_positives: bool = False,
        query_special_suffix: str = None,
        encoder_type: str = None,
    ):
        if selector:
            self.selector = hydra.utils.instantiate(selector)
        else:
            self.selector = DEFAULT_SELECTOR
        self.special_token = special_token
        self.encoder_type = encoder_type
        self.shuffle_positives = shuffle_positives
        self.query_special_suffix = query_special_suffix

    def load_data(self):
        raise NotImplementedError

    def __getitem__(self, index) -> BiEncoderSample:
        raise NotImplementedError

    def _process_query(self, query: str):
        # as of now, always normalize query
        query = normalize_question(query)
        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix

        return query


def get_dpr_files(source_name) -> List[str]:
    if os.path.exists(source_name) or glob.glob(source_name):
        return glob.glob(source_name)
    else:
        # try to use data downloader
        from dpr.data.download_data import download

        return download(source_name)
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

@app.add("csn_go_q")
def get_csn_go_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_go_qa")
def get_csn_go_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_go_a")
def get_csn_go_decomp(entry):
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


@app.add("csn_php_q")
def get_csn_php_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("csn_php_qa")
def get_csn_php_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("csn_php_a")
def get_csn_php_decomp(entry):
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




class EPRDataset(Dataset):
    def __init__(
        self,
        file: str,
        task_name,
        setup_type,
        top_k,
        loss_type=None,
        rank_loss_top_sample=2,
        rank_loss_factor=1,
        rank_candidate_num=8,
        ds_size=None,
        max_instances=None,
        hard_neg=False,
        selector: DictConfig = None,
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        normalize: bool = False,
        query_special_suffix: str = None,
    ):
        super().__init__(
            selector,
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix,
        )
        assert loss_type in ['epr','list_ranking']
        if loss_type == 'list_ranking':
            assert rank_candidate_num>3, 'EPRDataset.rank_candidate:{}'.format(rank_candidate_num)
        self.rank_candidate_num = rank_candidate_num
        self.loss_type = loss_type
        self.rank_loss_factor = rank_loss_factor
        self.rank_loss_top_sample = rank_loss_top_sample
        self.top_k = top_k
        self.max_instances=None
        self.task_name = task_name
        # self.file = f"dpr/{file.replace('.','/')}"
        self.file = file
        self.data_files = []
        self.hard_neg = hard_neg
        self.data = []
        self.normalize = normalize
        self.dataset = dataset_dict.functions[task_name]()
        self.setup_type = setup_type
        assert self.setup_type in ["q","qa","a"]
        self.get_field = app.functions[f"{self.task_name}_{self.setup_type}"]
        self.train_dataset = load_train_dataset(self.dataset,size=ds_size)
        # if self.max_instances is None and self.task_name=="smcalflow":
        #     self.max_instances = 44000

        logger.info("Data files: %s", self.data_files)

    def format_prompt(self, entry,is_train=True):
        # question = entry['question'] if "question" in entry else entry['question_text']
        # text = f"{question}\t{reformat(entry['decomposition'])}"
        if "id" in entry and entry['id'] is not None:
            entry = self.train_dataset[entry['id']]
        return {"title":"","text":self.get_field(entry),}



    def get_entry(self, entry):
        if self.loss_type == 'epr':
            positive_cntx = [self.format_prompt(p_example) for p_example in entry['ctxs'][:self.top_k]]
            negative_cntx = [self.format_prompt(n_example) for n_example in random.choices(self.train_dataset,k=50)]
            hard_negative_ctxs = [self.format_prompt(p_example) for p_example in entry['ctxs'][-self.top_k:]]
            if "question" in entry:
                question = entry['question']
            elif "question_text" in entry:
                question = entry['question_text']
            elif "user_utterance" in entry:
                question = entry['user_utterance']
            elif "document" in entry:
                question = entry['document']
            elif "src" in entry:
                question = entry['src']
            elif "source" in entry:
                question = entry['source']
            elif "translation.de" in entry:
                question = entry['translation.de']
            elif "input" in entry:
                question = entry['input']
            elif "joined_concepts" in entry:
                question = entry['joined_concepts']
            elif "article" in entry:
                question = entry['article']
            elif "sentence" in entry:
                question = entry['sentence']
            else:
                assert False
            entry = {"question":question,"answers":[],"positive_ctxs":positive_cntx,"negative_ctxs":negative_cntx}
            if self.hard_neg:
                entry["hard_negative_ctxs"] = hard_negative_ctxs
            return entry
        elif self.loss_type == 'list_ranking':

            if "question" in entry:
                question = entry['question']
            elif "question_text" in entry:
                question = entry['question_text']
            elif "user_utterance" in entry:
                question = entry['user_utterance']
            elif "document" in entry:
                question = entry['document']
            elif "src" in entry:
                question = entry['src']
            elif "source" in entry:
                question = entry['source']
            elif "translation.de" in entry:
                question = entry['translation.de']
            elif "input" in entry:
                question = entry['input']
            elif "joined_concepts" in entry:
                question = entry['joined_concepts']
            elif "article" in entry:
                question = entry['article']
            elif "sentence" in entry:
                question = entry['sentence']
            else:
                assert False

            sorted_cntx = []
            top_cntx_num = len(entry['ctxs'])
            ctxs_and_index = list(enumerate(entry['ctxs']))
            sorted_cntx.extend(random.sample(ctxs_and_index[:self.top_k],self.rank_loss_top_sample))
            sorted_cntx.extend(random.sample(ctxs_and_index[-self.top_k:],1))
            sorted_cntx.extend(random.sample(ctxs_and_index[self.top_k:-self.top_k],self.rank_candidate_num-1-self.rank_loss_top_sample))
            sorted_cntx.sort(key=lambda x:x[0])
            sorted_cntx = list(map(lambda x:x[1],sorted_cntx))
            # sorted_cntx.extend(random.sample(entry['ctxs'][:self.top_k],2))
            sorted_cntx = [self.format_prompt(ctx) for ctx in sorted_cntx]
            entry = {"question": question, "answers": [], "sorted_ctxs":sorted_cntx}
            return entry
            pass
        raise NotImplementedError

    def load_data(self):
        print(self.file)
        self.data_files = get_dpr_files(self.file)
        print(self.data_files)

        self.data = read_data_from_json_files(self.data_files)
        # if self.task_name=="smcalflow":
        if self.max_instances is not None:
            idx_list = list(range(len(self.data)))
            random.Random(42).shuffle(idx_list)
            self.data = [self.data[x] for x in idx_list[:self.max_instances]]
        # print(data[0])
        # filter those without positive ctx
        # self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total data size: {}".format(len(self.data)))

    def __getitem__(self, index) -> BiEncoderSample:
        if self.loss_type == 'epr':
            json_sample = self.get_entry(self.data[index])
            r = BiEncoderSample()
            r.query = self._process_query(json_sample["question"])

            positive_ctxs = json_sample["positive_ctxs"]
            negative_ctxs = (
                json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
            )
            hard_negative_ctxs = (
                json_sample["hard_negative_ctxs"]
                if "hard_negative_ctxs" in json_sample
                else []
            )

            for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
                if "title" not in ctx:
                    ctx["title"] = None

            def create_passage(ctx: dict):
                return BiEncoderPassage(
                    normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                    ctx["title"],
                )

            r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
            r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
            r.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
            return r
        elif self.loss_type == 'list_ranking':
            json_sample = self.get_entry(self.data[index])
            r = BiEncoderSample_for_list_ranking()
            r.query = self._process_query(json_sample['question'])

            def create_passage(ctx: dict):
                return BiEncoderPassage(
                    normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                    ctx["title"],
                )
            sorted_ctxs = json_sample['sorted_ctxs']
            r.sorted_passages = [create_passage(ctx) for ctx in sorted_ctxs]
            return r
            pass
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def get_qas(self) -> Tuple[List[str], List[str]]:
        return [s["question"] for s in self.data], [s["answers"] for s in self.data]

    def get_qas_range(
        self, start_idx: int, end_idx: int
    ) -> Tuple[List[str], List[str]]:
        return (
            [s["question"] for s in self.data[start_idx:end_idx]],
            [s["answers"] for s in self.data[start_idx:end_idx]],
        )



class JsonQADataset(Dataset):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        normalize: bool = False,
        query_special_suffix: str = None,
    ):
        super().__init__(
            selector,
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix,
        )
        # self.file = f"dpr/{file.replace('.','/')}"
        self.file = file
        self.data_files = []
        self.data = []
        self.normalize = normalize
        logger.info("Data files: %s", self.data_files)

    def load_data(self):
        print(self.file)
        self.data_files = get_dpr_files(self.file)
        print(self.data_files)

        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(self.data)))

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = self._process_query(json_sample["question"])

        positive_ctxs = json_sample["positive_ctxs"]
        negative_ctxs = (
            json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        )
        hard_negative_ctxs = (
            json_sample["hard_negative_ctxs"]
            if "hard_negative_ctxs" in json_sample
            else []
        )

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                ctx["title"],
            )

        r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        r.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
        return r

    def __len__(self):
        return len(self.data)

    def get_qas(self) -> Tuple[List[str], List[str]]:
        return [s["question"] for s in self.data], [s["answers"] for s in self.data]

    def get_qas_range(
        self, start_idx: int, end_idx: int
    ) -> Tuple[List[str], List[str]]:
        return (
            [s["question"] for s in self.data[start_idx:end_idx]],
            [s["answers"] for s in self.data[start_idx:end_idx]],
        )


def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    return ctx_text


def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question


class Cell:
    def __init__(self):
        self.value_tokens: List[str] = []
        self.type: str = ""
        self.nested_tables: List[Table] = []

    def __str__(self):
        return " ".join(self.value_tokens)

    def to_dpr_json(self, cell_idx: int):
        r = {"col": cell_idx}
        r["value"] = str(self)
        return r


class Row:
    def __init__(self):
        self.cells: List[Cell] = []

    def __str__(self):
        return "| ".join([str(c) for c in self.cells])

    def visit(self, tokens_function, row_idx: int):
        for i, c in enumerate(self.cells):
            if c.value_tokens:
                tokens_function(c.value_tokens, row_idx, i)

    def to_dpr_json(self, row_idx: int):
        r = {"row": row_idx}
        r["columns"] = [c.to_dpr_json(i) for i, c in enumerate(self.cells)]
        return r


class Table(object):
    def __init__(self, caption=""):
        self.caption = caption
        self.body: List[Row] = []
        self.key = None
        self.gold_match = False

    def __str__(self):
        table_str = "<T>: {}\n".format(self.caption)
        table_str += " rows:\n"
        for i, r in enumerate(self.body):
            table_str += " row #{}: {}\n".format(i, str(r))

        return table_str

    def get_key(self) -> str:
        if not self.key:
            self.key = str(self)
        return self.key

    def visit(self, tokens_function, include_caption: bool = False) -> bool:
        if include_caption:
            tokens_function(self.caption, -1, -1)
        for i, r in enumerate(self.body):
            r.visit(tokens_function, i)

    def to_dpr_json(self):
        r = {
            "caption": self.caption,
            "rows": [r.to_dpr_json(i) for i, r in enumerate(self.body)],
        }
        if self.gold_match:
            r["gold_match"] = 1
        return r


class NQTableParser(object):
    def __init__(self, tokens, is_html_mask, title):
        self.tokens = tokens
        self.is_html_mask = is_html_mask
        self.max_idx = len(self.tokens)
        self.all_tables = []

        self.current_table: Table = None
        self.tables_stack = collections.deque()
        self.title = title

    def parse(self) -> List[Table]:
        self.all_tables = []
        self.tables_stack = collections.deque()

        for i in range(self.max_idx):

            t = self.tokens[i]

            if not self.is_html_mask[i]:
                # cell content
                self._on_content(t)
                continue

            if "<Table" in t:
                self._on_table_start()
            elif t == "</Table>":
                self._on_table_end()
            elif "<Tr" in t:
                self._onRowStart()
            elif t == "</Tr>":
                self._onRowEnd()
            elif "<Td" in t or "<Th" in t:
                self._onCellStart()
            elif t in ["</Td>", "</Th>"]:
                self._on_cell_end()

        return self.all_tables

    def _on_table_start(self):
        caption = self.title
        parent_table = self.current_table
        if parent_table:
            self.tables_stack.append(parent_table)

            caption = parent_table.caption
            if parent_table.body and parent_table.body[-1].cells:
                current_cell = self.current_table.body[-1].cells[-1]
                caption += " | " + " ".join(current_cell.value_tokens)

        t = Table()
        t.caption = caption
        self.current_table = t
        self.all_tables.append(t)

    def _on_table_end(self):
        t = self.current_table
        if t:
            if self.tables_stack:  # t is a nested table
                self.current_table = self.tables_stack.pop()
                if self.current_table.body:
                    current_cell = self.current_table.body[-1].cells[-1]
                    current_cell.nested_tables.append(t)
        else:
            logger.error("table end without table object")

    def _onRowStart(self):
        self.current_table.body.append(Row())

    def _onRowEnd(self):
        pass

    def _onCellStart(self):
        current_row = self.current_table.body[-1]
        current_row.cells.append(Cell())

    def _on_cell_end(self):
        pass

    def _on_content(self, token):
        if self.current_table.body:
            current_row = self.current_table.body[-1]
            current_cell = current_row.cells[-1]
            current_cell.value_tokens.append(token)
        else:  # tokens outside of row/cells. Just append to the table caption.
            self.current_table.caption += " " + token


def read_nq_tables_jsonl(path: str) -> Dict[str, Table]:
    tables_with_issues = 0
    single_row_tables = 0
    nested_tables = 0
    regular_tables = 0
    total_tables = 0
    total_rows = 0
    tables_dict = {}

    with jsonlines.open(path, mode="r") as jsonl_reader:
        for jline in jsonl_reader:
            tokens = jline["tokens"]

            if "( hide ) This section has multiple issues" in " ".join(tokens):
                tables_with_issues += 1
                continue

            mask = jline["html_mask"]
            # page_url = jline["doc_url"]
            title = jline["title"]
            p = NQTableParser(tokens, mask, title)
            tables = p.parse()

            # table = parse_table(tokens, mask)

            nested_tables += len(tables[1:])

            for t in tables:
                total_tables += 1

                # calc amount of non empty rows
                non_empty_rows = sum(
                    [
                        1
                        for r in t.body
                        if r.cells and any([True for c in r.cells if c.value_tokens])
                    ]
                )

                if non_empty_rows <= 1:
                    single_row_tables += 1
                else:
                    regular_tables += 1
                    total_rows += len(t.body)

                    if t.get_key() not in tables_dict:
                        tables_dict[t.get_key()] = t

            if len(tables_dict) % 1000 == 0:
                logger.info("tables_dict %d", len(tables_dict))

    logger.info("regular tables %d", regular_tables)
    logger.info("tables_with_issues %d", tables_with_issues)
    logger.info("single_row_tables %d", single_row_tables)
    logger.info("nested_tables %d", nested_tables)
    return tables_dict


def get_table_string_for_answer_check(table: Table):  # this doesn't use caption
    table_text = ""
    for r in table.body:
        table_text += " . ".join([" ".join(c.value_tokens) for c in r.cells])
    table_text += " . "
    return table_text


class JsonLTablesQADataset(Dataset):
    def __init__(
        self,
        file: str,
        is_train_set: bool,
        selector: DictConfig = None,
        shuffle_positives: bool = False,
        max_negatives: int = 1,
        seed: int = 0,
        max_len=100,
        split_type: str = "type1",
    ):
        super().__init__(selector, shuffle_positives=shuffle_positives)
        self.data_files = glob.glob(file)
        self.data = []
        self.is_train_set = is_train_set
        self.max_negatives = max_negatives
        self.rnd = random.Random(seed)
        self.max_len = max_len
        self.linearize_func = JsonLTablesQADataset.get_lin_func(split_type)

    def load_data(self):
        data = []
        for path in self.data_files:
            with jsonlines.open(path, mode="r") as jsonl_reader:
                data += [jline for jline in jsonl_reader]

        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(self.data)))

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = json_sample["question"]
        positive_ctxs = json_sample["positive_ctxs"]
        hard_negative_ctxs = json_sample["hard_negative_ctxs"]

        if self.shuffle_positives:
            self.rnd.shuffle(positive_ctxs)

        if self.is_train_set:
            self.rnd.shuffle(hard_negative_ctxs)
        positive_ctxs = positive_ctxs[0:1]
        hard_negative_ctxs = hard_negative_ctxs[0 : self.max_negatives]

        r.positive_passages = [
            BiEncoderPassage(self.linearize_func(self, ctx, True), ctx["caption"])
            for ctx in positive_ctxs
        ]
        r.negative_passages = []
        r.hard_negative_passages = [
            BiEncoderPassage(self.linearize_func(self, ctx, False), ctx["caption"])
            for ctx in hard_negative_ctxs
        ]
        return r

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_lin_func(cls, split_type: str):
        f = {
            "type1": JsonLTablesQADataset._linearize_table,
        }
        return f[split_type]

    @classmethod
    def split_table(cls, t: dict, max_length: int):
        rows = t["rows"]
        header = None
        header_len = 0
        start_row = 0

        # get the first non empty row as the "header"
        for i, r in enumerate(rows):
            row_lin, row_len = JsonLTablesQADataset._linearize_row(r)
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                header = row_lin
                header_len += row_len
                start_row = i
                break

        chunks = []
        current_rows = [header]
        current_len = header_len

        for i in range(start_row + 1, len(rows)):
            row_lin, row_len = JsonLTablesQADataset._linearize_row(rows[i])
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                current_rows.append(row_lin)
                current_len += row_len
            if current_len >= max_length:
                # linearize chunk
                linearized_str = "\n".join(current_rows) + "\n"
                chunks.append(linearized_str)
                current_rows = [header]
                current_len = header_len

        if len(current_rows) > 1:
            linearized_str = "\n".join(current_rows) + "\n"
            chunks.append(linearized_str)
        return chunks

    def _linearize_table(self, t: dict, is_positive: bool) -> str:
        rows = t["rows"]
        selected_rows = set()
        rows_linearized = []
        total_words_len = 0

        # get the first non empty row as the "header"
        for i, r in enumerate(rows):
            row_lin, row_len = JsonLTablesQADataset._linearize_row(r)
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                selected_rows.add(i)
                rows_linearized.append(row_lin)
                total_words_len += row_len
                break

        # split to chunks
        if is_positive:
            row_idx_with_answers = [ap[0] for ap in t["answer_pos"]]

            if self.shuffle_positives:
                self.rnd.shuffle(row_idx_with_answers)
            for i in row_idx_with_answers:
                if i not in selected_rows:
                    row_lin, row_len = JsonLTablesQADataset._linearize_row(rows[i])
                    selected_rows.add(i)
                    rows_linearized.append(row_lin)
                    total_words_len += row_len
                if total_words_len >= self.max_len:
                    break

        if total_words_len < self.max_len:  # append random rows

            if self.is_train_set:
                rows_indexes = np.random.permutation(range(len(rows)))
            else:
                rows_indexes = [*range(len(rows))]

            for i in rows_indexes:
                if i not in selected_rows:
                    row_lin, row_len = JsonLTablesQADataset._linearize_row(rows[i])
                    if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                        selected_rows.add(i)
                        rows_linearized.append(row_lin)
                        total_words_len += row_len
                    if total_words_len >= self.max_len:
                        break

        linearized_str = ""
        for r in rows_linearized:
            linearized_str += r + "\n"

        return linearized_str

    @classmethod
    def _linearize_row(cls, row: dict) -> Tuple[str, int]:
        cell_values = [c["value"] for c in row["columns"]]
        total_words = sum(len(c.split(" ")) for c in cell_values)
        return ", ".join([c["value"] for c in row["columns"]]), total_words


def split_tables_to_chunks(
    tables_dict: Dict[str, Table], max_table_len: int, split_type: str = "type1"
) -> List[Tuple[int, str, str, int]]:
    tables_as_dicts = [t.to_dpr_json() for k, t in tables_dict.items()]
    chunks = []
    chunk_id = 0
    for i, t in enumerate(tables_as_dicts):
        # TODO: support other types
        assert split_type == "type1"
        table_chunks = JsonLTablesQADataset.split_table(t, max_table_len)
        title = t["caption"]
        for c in table_chunks:
            # chunk id , text, title, external_id
            chunks.append((chunk_id, c, title, i))
            chunk_id += 1
        if i % 1000 == 0:
            logger.info("Splitted %d tables to %d chunks", i, len(chunks))
    return chunks
