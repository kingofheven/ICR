
import argparse
import copy
import json
import os.path

from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk


def convert_kp20k():
    print("input_file: ", args.input_file)
    print("input_scored_file: ", args.input_scored_file)
    print("output_file: ", args.output_file)
    print("output_dep_file: ", args.output_dep_file)

    with open(args.input_scored_file, "r") as data_file:
        input_scored_data = json.load(data_file)

    with open(args.input_file, "r") as data_file:
        input_data = json.load(data_file)
    print("len of input file: ", len(input_data))
    print("len of input scored file", len(input_scored_data))
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
    dataset = load_dataset(dataset_path[args.dataset])

    train_set = dataset[args.split]

    out_data = []
    out_dep_data = []
    print("len of train set: ", len(train_set))
    idx_attr = "idx"
    idx2scored = {e[idx_attr]: e['ctxs'] for e in input_scored_data}
    # for i in tqdm(range(len(input_scored_data))):
    for i in tqdm(range(len(input_data))):
        idx = train_set[i][idx_attr]
        out_data.append(copy.deepcopy(train_set[i]))
        out_dep_data.append(copy.deepcopy(train_set[i]))
        ctx = [{'id': e['id']} for e in input_data[i]['ctxs'] if e['id'] != idx]

        if idx in idx2scored:
            ctx_in_scored = idx2scored[idx]
        else:
            ctx_in_scored = []



        ctx_in_scored = [dict(t) for t in set([tuple(d.items()) for d in ctx_in_scored])]
        ctx_in_scored.sort(key=lambda x: x['score'])


        if args.split == "train":
            id_in_scored = [x['id'] for x in ctx_in_scored if x['id'] != i]
            id_in_rtv = [x['id'] for x in input_data[i]['ctxs'] if x['id'] != i]
        else:
            id_in_scored = [x['id'] for x in ctx_in_scored if x['id'] != idx]
            id_in_rtv = [x['id'] for x in input_data[i]['ctxs'] if x['id'] != idx]
        out_data[i]['ctxs'] = [e for e in ctx if e['id'] not in id_in_scored]
        if args.num == 'all':
            out_dep_data[i]['ctxs'] = [e for e in ctx_in_scored]
        elif args.num == "static":
            out_dep_data[i]['ctxs'] = [e for e in ctx_in_scored if e['id'] in id_in_rtv or e['score'] <= ctx_in_scored[4]['score']]
        else:
            raise NotImplementedError

    print("len of out file: ", len(out_data))
    print("len of out dep file: ", len(out_dep_data))
    with open(args.output_file, "w") as data_file:
        json.dump(out_data, data_file)
    with open(args.output_dep_file, "w") as data_file:
        json.dump(out_dep_data, data_file)

def convert():
    print("input_file: ", args.input_file)
    print("input_scored_file: ", args.input_scored_file)
    print("output_file: ", args.output_file)
    print("output_dep_file: ", args.output_dep_file)

    with open(args.input_scored_file, "r") as data_file:
        input_scored_data = json.load(data_file)

    with open(args.input_file, "r") as data_file:
        input_data = json.load(data_file)
    print("len of input file: ", len(input_data))
    print("len of input scored file", len(input_scored_data))



    train_set = dataset[args.split]

    out_data = []
    out_dep_data = []
    offset = 0
    for i in tqdm(range(len(train_set))):
        if train_set[i][question].replace("'", "").replace("’", "") != input_data[i]['question'].replace("'", "").replace("’", ""):
            print("different question id: ", i)
            print(train_set[i][question])
            print(input_data[i]['question'])
        out_data.append(copy.deepcopy(train_set[i]))
        out_dep_data.append(copy.deepcopy(train_set[i]))

        ctx = [{'id': e['id']} for e in input_data[i]['ctxs'] if e['id'] != i]

        if train_set[i][idx] != input_scored_data[i-offset][idx]:
            out_data[i]['ctxs'] = ctx
            out_dep_data[i]['ctxs'] = []
            offset += 1
        else:

            ctx_in_scored = input_scored_data[i - offset]['ctxs']
            ctx_in_scored = [dict(t) for t in set([tuple(d.items()) for d in ctx_in_scored])]
            ctx_in_scored.sort(key=lambda x: x['score'])

            id_in_scored = [x['id'] for x in ctx_in_scored if x['id'] != i]
            id_in_rtv = [x['id'] for x in input_data[i]['ctxs'] if x['id'] != i]
            out_data[i]['ctxs'] = [e for e in ctx if e['id'] not in id_in_scored]
            if args.num == 'all':
                out_dep_data[i]['ctxs'] = [e for e in ctx_in_scored]
            elif args.num == "static":
                out_dep_data[i]['ctxs'] = [e for e in ctx_in_scored if
                                           e['id'] in id_in_rtv or e['score'] <= ctx_in_scored[4]['score']]
            else:
                raise NotImplementedError


    print("len of out file: ", len(out_data))
    print("len of out dep file: ", len(out_dep_data))
    with open(args.output_file, "w") as data_file:
        json.dump(out_data, data_file)
    with open(args.output_dep_file, "w") as data_file:
        json.dump(out_dep_data, data_file)


def merge():
    print("scored_file1: ", args.scored_file1)
    print("scored_file2: ", args.scored_file2)
    print("output_file: ", args.output_file)

    with open(args.scored_file1, "r") as data_file:
        scored_data1 = json.load(data_file)
    with open(args.scored_file2, "r") as data_file:
        scored_data2 = json.load(data_file)

    print(len(scored_data1), len(scored_data2))
    idx = 'idx'

    idx2sc2 = {e[idx]: e['ctxs'] for e in scored_data2}
    for i in tqdm(scored_data1):  
        idx_i = i[idx]
        if idx_i in idx2sc2:
            i['ctxs'] += idx2sc2[idx_i]


        i['ctxs'] = [dict(t) for t in set([tuple(d.items()) for d in i['ctxs']])]


        i['ctxs'].sort(key=lambda x: x['score'])

    out_data = [e for e in scored_data1 if e['ctxs']]
    print("len of merged data: ", len(out_data))

    with open(args.output_file, "w") as data_file:
        json.dump(out_data, data_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, default="convert")
    parser.add_argument("--dataset", type=str, default="mtop")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num", type=str, default="static")
    parser.add_argument("--input_file", type=str, default=".")
    parser.add_argument("--input_scored_file", type=str, default=".")
    parser.add_argument("--output_file", type=str, default=".")
    parser.add_argument("--output_dep_file", type=str, default=".")
    parser.add_argument("--scored_file1", type=str, default=".")
    parser.add_argument("--scored_file2", type=str, default=".")
    args = parser.parse_args()

    if args.func == "convert":
        if args.dataset not in ["mtop", "break"]:
            convert_kp20k()
        else:
            convert()
    elif args.func == "merge":
        merge()
