import os
from Channel_LM_Prompting.util import get_prompts, get_label_from_template, get_one_prompt
from copy import deepcopy
from tqdm import tqdm


def get_test_labels(data, task, idx):
    ret = []
    test_labels = get_prompts(task, idx)
    for e in data:
        for l in test_labels:
            tmp = deepcopy(e)
            tmp['test_label'] = l
            ret.append(tmp)
    return ret

def change_prompt_template(data, task, idx):
    for e in tqdm(data):
        for ctx in e['ctxs']:
            spt = ctx['text'].split('\t')
            q = "\t".join(spt[:-1])
            a = spt[-1]
            label = get_label_from_template(task, a)
            new_a = get_one_prompt(task, idx, label)
            ctx['text'] = q + '\t' + new_a
    return data


def get_multi_choice_labels(data, task, split):
    ret = []
    dataset_path = {
        "concode": "mengmengmmm/concode_trainuse",
        "java2cs": "mengmengmmm/java2cs_trainuse",
        "csn_ruby": "mengmengmmm/csn_ruby_trainuse",
        "csn_python": "mengmengmmm/csn_python_trainuse",
        "csn_php": "mengmengmmm/csn_php_trainuse",
        "csn_js": "mengmengmmm/csn_js_trainuse",
        "csn_go": "mengmengmmm/csn_go_trainuse",
        "csn_go_slice1": "mengmengmmm/csn_go_trainuse_slice1",
        "csn_go_slice2": "mengmengmmm/csn_go_trainuse_slice2",
        "csn_go_slice3": "mengmengmmm/csn_go_trainuse_slice3",
        "csn_go_slice4": "mengmengmmm/csn_go_trainuse_slice4",
        "csn_python_slice1": "mengmengmmm/csn_python_trainuse_slice1",
        
        "tlc": "mengmengmmm/tlc",
        "tlc_slice1": "mengmengmmm/tlc_slice1",
        "tlc_slice2": "mengmengmmm/tlc_slice2",
        
        "csn_java": "mengmengmmm/csn_java",
        "csn_java_slice1": "mengmengmmm/csn_java_slice1",
        "csn_java_slice2": "mengmengmmm/csn_java_slice2",
        "csn_java_slice3": "mengmengmmm/csn_java_slice3",
        "csn_java_slice4": "mengmengmmm/csn_java_slice4",
        "conala": "mengmengmmm/conala",
        "b2f_medium": "mengmengmmm/B2F_medium",
        "b2f_small": "mengmengmmm/B2F_small",
        

    }
    from datasets import load_dataset
    ds = load_dataset(dataset_path[task])
    q_to_choices = {}
    for e in ds[split]:
        q_to_choices[e['question'].replace("’", "").replace("'", "")] = e['choices']
    for e in tqdm(data):
        if 'choices' not in e:
            k = e['question'].replace("’", "").replace("'", "")  # ’
            e['choices'] = q_to_choices[k]
        e['choices'] = e['choices'].split('\n')
        for choice in e['choices']:
            tmp = deepcopy(e)
            tmp['test_label'] = choice
            ret.append(tmp)
    return ret
