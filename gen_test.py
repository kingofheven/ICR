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



import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('punkt')

def calculate_meteor_score(reference_list, hypothesis_list):

    if len(reference_list) != len(hypothesis_list):
        raise ValueError("ERROR")

    scores = []
    for reference, hypothesis in zip(reference_list, hypothesis_list):
        reference_tokens = word_tokenize(reference)
        hypothesis_tokens = word_tokenize(hypothesis)
        score = meteor_score([reference_tokens], hypothesis_tokens)
        scores.append(score)

    average_score = sum(scores) / len(scores)
    return average_score



import sacrebleu


def calculate_chrf(reference_list, hypothesis_list):

    if len(reference_list) != len(hypothesis_list):
        raise ValueError("ERROR")


    references = [reference_list]


    score = sacrebleu.corpus_chrf(hypothesis_list, references)

    return score.score



from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk





def get_synonyms(word):

    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def modified_bleu(reference, hypothesis):

    reference_tokens = word_tokenize(reference)
    hypothesis_tokens = word_tokenize(hypothesis)


    smooth_func = SmoothingFunction().method1
    base_bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smooth_func)


    synonym_matches = 0
    for token in hypothesis_tokens:
        synonyms = get_synonyms(token)
        if any(syn in reference_tokens for syn in synonyms):
            synonym_matches += 1

    synonym_score = synonym_matches / len(hypothesis_tokens) if hypothesis_tokens else 0


    final_score = (base_bleu_score + synonym_score) / 2
    return final_score


def calculate_crystal_bleu_scores(reference_list, hypothesis_list):

    if len(reference_list) != len(hypothesis_list):
        raise ValueError("ERROR")

    scores = []
    for reference, hypothesis in zip(reference_list, hypothesis_list):
        score = modified_bleu(reference, hypothesis)
        scores.append(score)

    average_score = sum(scores) / len(scores)
    return average_score


def renorm(text):
    text = text.split("\n")[0]
    text = re.sub("[\d]+\#\) ",";", text)
    return text

import argparse




def test_bleu(file_name):
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
            
                average_score = calculate_meteor_score([" ".join(i[0]) for i in ref_list], [" ".join(i) for i in hyp_list])
                print(f"Average METEOR Score: {float(average_score):.4f}")
                
                chrf_score = calculate_chrf([" ".join(i[0]) for i in ref_list], [" ".join(i) for i in hyp_list])
                print(f"Average chrF Score: {float(chrf_score):.4f}")
                
                
            if args.dataset in ['b2f_small','b2f_medium']:
                crystal_bleu_score = calculate_crystal_bleu_scores([" ".join(i[0]) for i in ref_list], [" ".join(i) for i in hyp_list])
                print(f"Average CrystalBLEU Score: {float(crystal_bleu_score):.4f}")    
                
            if args.dataset in ['conala']:
                from codebleu import calc_codebleu

                codebleu = calc_codebleu([" ".join(i[0]) for i in ref_list], [" ".join(i) for i in hyp_list], lang="python", weights=(0.25, 0.25, 0.25, 0.25))
                print('codebleu score:{}'.format(codebleu))
                print('codebleu score:{}'.format(codebleu['codebleu']))
                
                crystal_bleu_score = calculate_crystal_bleu_scores([" ".join(i[0]) for i in ref_list], [" ".join(i) for i in hyp_list])
                print(f"Average CrystalBLEU Score: {float(crystal_bleu_score):.4f}")
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














from qdecomp_with_dependency_graphs.scripts.eval.evaluate_predictions import evaluate
from qdecomp_with_dependency_graphs.scripts.eval.evaluate_predictions import format_qdmr




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
        test_bleu(tmp_fp)

    else:
        raise NotImplementedError

    fitlog.finish()  # finish the logging

