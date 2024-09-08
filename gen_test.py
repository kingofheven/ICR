import os
from src.utils.app import App
import json
import pandas as pd
from src.utils.eval_many import eval_many, eval_many_mtop,eval_many_smcalflow
from src.utils.cache_util import BufferedJsonWriter,BufferedJsonReader
import requests
import re
import numpy as np
import httpx
import asyncio
import torch
import fitlog

def renorm(text):
    text = text.split("\n")[0]
    text = re.sub("[\d]+\#\) ",";", text)
    return text

import argparse

def dwiki_bleu(file_name):
    from tqdm import tqdm
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.tokenize import word_tokenize
    import re, string
    ref_list = []
    hyp_list = []
    with open(file_name) as f:
        data = json.load(f)
    punctuation = '[%s]+' % re.escape(string.punctuation)
    for line in tqdm(data):
        ref = line['tgt'] if 'tgt' in line else line['answers'][0]
        hyp = line['generated'].split("<|endoftext|>")[0].strip()

        ref = re.sub(punctuation, '', ref)
        hyp = re.sub(punctuation, '', hyp)
        ref_list.append([word_tokenize(ref)])
        hyp_list.append(word_tokenize(hyp))
    print('bleu score:{}'.format(corpus_bleu(ref_list, hyp_list)))


def wikiauto_bleu(file_name):
    from tqdm import tqdm
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.tokenize import word_tokenize
    import re, string
    ref_list = []
    hyp_list = []
    with open(file_name) as f:
        data = json.load(f)
    punctuation = '[%s‘]+' % re.escape(string.punctuation)

    if args.dataset == 'wikiauto' and args.split not in ['debug', 'test_wiki']:
        from datasets import load_dataset
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
            "csn_java_slice1": "mengmengmmm/csn_java_slice1",
            "tlc": "mengmengmmm/tlc",
            "tlc_slice1": "mengmengmmm/tlc_slice1",
            "tlc_slice2": "mengmengmmm/tlc_slice2",
            
            "csn_java": "mengmengmmm/csn_java",
            "conala": "mengmengmmm/conala",
            "b2f_medium": "mengmengmmm/B2F_medium",
            "b2f_small": "mengmengmmm/B2F_small",
            
            
            
        }
        dataset = load_dataset(dataset_path[args.dataset])
        ref_dict = {}
        for e in dataset[args.split]:
            k= e['target']
            k = re.sub(punctuation, '', k)
            ref_dict[k] = e['references']

    if args.dataset in ['common_gen', 'opusparcus', 'squadv2', 'e2e', 'dart', 'totto']:
        from datasets import load_dataset
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
            "csn_java_slice1": "mengmengmmm/csn_java_slice1",
            "tlc": "mengmengmmm/tlc",
            "tlc_slice1": "mengmengmmm/tlc_slice1",
            "tlc_slice2": "mengmengmmm/tlc_slice2",
            
            "csn_java": "mengmengmmm/csn_java",
            "conala": "mengmengmmm/conala",
            "b2f_medium": "mengmengmmm/B2F_medium",
            "b2f_small": "mengmengmmm/B2F_small",
            
            
            
            

        }
        dataset = load_dataset(dataset_path[args.dataset])
        ref_dict = {}
        for e in dataset[args.split]:
            k= e['target']
            k = re.sub(punctuation, '', k)
            ref_dict[k] = e['references']
    q = []

    for line in tqdm(data):




        if args.dataset in ['tlc','tlc_slice1','csn_python','csn_java','csn_java_slice1','b2f_medium','b2f_small','conala','csn_go','csn_ruby','csn_js','csn_php']:
            ref = line['target'] if 'target' in line else line['answers'][0]
            hyp = line['generated'].split("\n")[0].strip()

            if type(ref)==list:
                ref=ref[0]

            ref_list.append([word_tokenize(ref)])
            hyp_list.append(word_tokenize(hyp))


    if args.dataset not in ['debug']:
        if args.dataset in ['tlc', 'tlc_slice1','b2f_medium','b2f_small','conala','csn_java','csn_java_slice1','csn_python','csn_go','csn_php','csn_js','csn_ruby']:
            bleu = round(corpus_bleu(ref_list, hyp_list, [1, 0, 0, 0]), 4)
            fitlog.add_best_metric({args.split: {'bleu1': bleu}})
            print('bleu1 score:{}'.format(bleu))
            bleu = round(corpus_bleu(ref_list, hyp_list, [0.5, 0.5, 0, 0]), 4)
            fitlog.add_best_metric({args.split: {'bleu2': bleu}})
            print('bleu2 score:{}'.format(bleu))
            bleu = round(corpus_bleu(ref_list, hyp_list, [0.25, 0.25, 0.25, 0.25]), 4)
            fitlog.add_best_metric({args.split: {'bleu4': bleu}})
            print('bleu4 score:{}'.format(bleu))
            

            if args.dataset in ["csn_java",'csn_python','csn_js','csn_php','csn_ruby']:

    
                from rouge_score import rouge_scorer
                import numpy as np
                
                def calculate_rouge_l_f1(references, hypotheses):
                    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                    
                    fmeasures = []
                    
                    for reference, hypothesis in zip(references, hypotheses):
                        scores = scorer.score(reference, hypothesis)
                        rouge_l_score = scores['rougeL']
                        fmeasures.append(rouge_l_score.fmeasure)
                    
                    # Calculate average F1 score
                    avg_fmeasure = np.mean(fmeasures)
                    
                    return avg_fmeasure



                rouge_l_score = calculate_rouge_l_f1([" ".join(i[0]) for i in ref_list], [" ".join(i) for i in hyp_list])
                print(f"Average ROUGE-L F1 Score: {float(rouge_l_score):.4f}")
            
            if args.dataset in ['conala']:
                from codebleu import calc_codebleu

                codebleu = calc_codebleu([" ".join(i[0]) for i in ref_list], [" ".join(i) for i in hyp_list], lang="python", weights=(0.25, 0.25, 0.25, 0.25))
                print('codebleu score:{}'.format(codebleu))
                print('codebleu score:{}'.format(codebleu['codebleu']))
        else:
            bleu = round(corpus_bleu(ref_list, hyp_list), 4)
            fitlog.add_best_metric({args.split: {'bleu': bleu}})
            print('bleu score:{}'.format(bleu))
    else:
        from rouge import Rouge
        rouge = Rouge()
        scores = rouge.get_scores(hyp_list, ref_list, avg=True)
        fitlog.add_best_metric({args.split: {'rouge-1': scores['rouge-1']['f'], 'rouge-2': scores['rouge-2']['f'],
                                             'rouge-l': scores['rouge-l']['f']}})
        print('rouge-1 score:{}'.format(scores['rouge-1']['f']))
        print('rouge-2 score:{}'.format(scores['rouge-2']['f']))
        print('rouge-l score:{}'.format(scores['rouge-l']['f']))



def kp20k_f1(file_name, split):
    from evaluate_prediction import main, ARGS
    import re
    from tqdm import tqdm
    with open(file_name, 'r') as f:
        data = json.load(f)
    src_list = []
    trg_list = []
    pred_list = []
    opt = ARGS()
    for i in tqdm(data):

        try:
            d = i['document'] if "document" in i else i['question']
            src_list.append(d)
            if split == 'abstract':
                if "abstractive_keyphrases" in i:
                    i['abstractive_keyphrases'] = eval(i['abstractive_keyphrases'])
                    trg_list.append(";".join(i['abstractive_keyphrases']))

                else:
                    i['abstractive_keyphrases'] = eval(i['answers'][0])
                    trg_list.append(";".join(i['abstractive_keyphrases']))
            else:
                if "extractive_keyphrases" in i:
                    i['extractive_keyphrases'] = eval(i['extractive_keyphrases'])
                    trg_list.append(";".join(i['extractive_keyphrases']))

                else:
                    i['extractive_keyphrases'] = eval(i['answers'][0])
                    trg_list.append(";".join(i['extractive_keyphrases']))


            pred = '"' + i['generated']
            reg = re.compile(r'"(.*?)"')
            pred = re.findall(reg, pred)
            if len(pred) == 0:
                print(i['generated'])
            pred_list.append(";".join(pred))
        except Exception as e:
            print(e)

    result_dict = main(opt,src_list,trg_list,pred_list)
    for k,v in result_dict.items():
        if '@M' in k and 'macro' in k:
            print('{} : {}'.format(k,v))

def add_mtop_acc(file_name,id_list=None):
    correct_list = []
    with open(file_name) as f:
        line_list = []
        data = json.load(f)

        for line in data:
            if id_list is not None and line['id'] not in id_list:
                continue
            lf = line['logical_form'] if 'logical_form' in line else line['answers'][0]
            line_list.append((line['generated'].split("<|endoftext|>")[0].strip(),lf))
    pred,gold = list(zip(*line_list))
    res_list = eval_many_mtop(pred,gold)

    for entry,acc in zip(data,res_list):
        entry['acc'] =acc


    res_list_int = list(map(int,res_list))

    acc_result = sum(res_list_int)/len(res_list_int)
    print('file_name:{}'.format(file_name))
    print('prediction acc:{}'.format(acc_result))
    fitlog.add_best_metric({args.split: {'acc': acc_result}})
    return data


def add_spider_acc(file_name,id_list=None):
    from datasets import load_from_disk
    import spider.evaluation
    from tqdm import tqdm
    with open(file_name, 'r') as f:
        input_data = json.load(f)
    ds = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/spider")

    db_dict = {}
    for e in ds['validation']:
        db_dict[e['query']] = e['db_id']

    pred_list = []
    gold_list = []
    num = 0
    for e in tqdm(input_data):
        p = e['generated'].split("<|endoftext|>")[0].strip()
        g = e['query'] if 'query' in e else e['answers'][0]
        db_id = db_dict[g]

        if p == g:
            num += 1
        g = [g, db_id]
        if len(p) == 0:
            p = "a"
        p = [p, db_id]
        pred_list.append(p)
        gold_list.append(g)
    ret = spider.evaluation.evaluate_in_memory(pred_list, gold_list)
    fitlog.add_best_metric({args.split: {'acc': ret['total_scores']['all']['exact']}})

def add_break_acc(path, id_list=None):
    with BufferedJsonReader(path) as f:
        df = pd.DataFrame(f.read())
    data = df.to_dict("records")
    question_field = "question" if "question" in data[0] else 'question_text'
    zipped_data = []
    for entry in data:
        if id_list is not None and entry['id'] in id_list:
            continue
        generated = renorm(entry['generated'].split("\n")[0].split("<|endoftext|>")[0]).strip()
        decomposition = entry['decomposition'] if "decomposition" in entry else entry['answers'][0]

        zipped_data.append([entry[question_field], generated, decomposition])

    questions, pred, gold = list(zip(*zipped_data))
    acc_results = eval_many(questions, pred, gold)
    acc_results_int = list(map(int,acc_results))
    acc_result = sum(acc_results_int)/len(acc_results_int)
    print('result:')
    print(acc_result)
    fitlog.add_best_metric({args.split: {'acc': acc_result}})

    for entry, acc in zip(data, acc_results):
        entry['acc'] = acc
    return data


def add_smcalflow_acc(file_name,id_list=None):
    correct_list = []
    with open(file_name) as f:
        data = json.load(f)
        line_list = []
        for line in data:
            if id_list is not None and line['id'] not in id_list:
                continue
            lf = line['lispress'] if 'lispress' in line else line['answers'][0]
            line_list.append((line['generated'].split("<|endoftext|>")[0].strip(),lf))
    pred,gold = list(zip(*line_list))
    res_list = eval_many_smcalflow(pred,gold)

    acc_results_int = list(map(int,res_list))
    acc_result = sum(acc_results_int)/len(acc_results_int)
    print('result:')
    print(acc_result)
    fitlog.add_best_metric({args.split: {'acc': acc_result}})

    for entry,acc in zip(data,res_list):
        entry['acc'] =acc
    return data

from qdecomp_with_dependency_graphs.scripts.eval.evaluate_predictions import evaluate
from qdecomp_with_dependency_graphs.scripts.eval.evaluate_predictions import format_qdmr
def eval_break_test(file_name,id_list=None):
    with BufferedJsonReader(file_name) as f:
        df = pd.DataFrame(f.read())
    data = df.to_dict("records")
    question_field = "question" if "question" in data[0] else 'question_text'
    zipped_data = []
    for entry in data:
        if id_list is not None and entry['id'] in id_list:
            continue
        generated = renorm(entry['generated'].split("\n")[0].split("<|endoftext|>")[0]).strip()
        decomposition = entry['decomposition'] if "decomposition" in entry else entry['answers'][0]

        zipped_data.append([entry[question_field], generated, decomposition, entry['question_id']])

    questions, predictions, golds, question_ids = list(zip(*zipped_data))

    predictions = [format_qdmr(pred.replace("  "," ")) for pred in predictions]
    golds = [format_qdmr(gold) for gold in golds]

    res = evaluate(question_ids=question_ids,
                   questions=questions,
                   golds=golds,
                   decompositions=predictions,
                   metadata=None,
                   output_path_base=None,
                   num_processes=8)

    print(res)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',type=str)
    parser.add_argument('--split',type=str,default="test")
    parser.add_argument('--fp',)
    parser.add_argument('--exp_name', type=str)

    parser.add_argument('--method', type=str)
    parser.add_argument('--plm', type=str)
    parser.add_argument('--iter_scored_num', type=str)
    parser.add_argument('--iter_num', type=str)
    parser.add_argument('--epoch_num', type=str, default="10")
    parser.add_argument('--prompt_num', type=str)
    parser.add_argument('--alpha', type=str)
    parser.add_argument('--beilv', type=str)


    args = parser.parse_args()

    fitlog.set_log_dir("icr_fitlog/metric_logs/")  
    fitlog.add_hyper(args)  

   
    if args.fp==None:
        tmp_fp = 'data/bm25_{}_result_{}.json'.format(args.dataset,args.split)
    else:
        tmp_fp = args.fp

    

    if args.dataset in ['tlc','tlc_slice1','csn_go','csn_ruby','csn_js','csn_php','b2f_medium','b2f_small','conala','csn_java_slice1','csn_java','csn_python']:
        wikiauto_bleu(tmp_fp)

    else:
        raise NotImplementedError

    fitlog.finish()  # finish the logging

