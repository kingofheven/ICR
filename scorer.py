import torch
import tqdm
from torch.utils.data import DataLoader
from src.data.collators import DataCollatorWithPaddingAndCuda
import hydra.utils as hu 
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import json
import os
from omegaconf import OmegaConf

from allennlp.nn.util import sequence_cross_entropy_with_logits,get_mask_from_sequence_lengths
from transformers.data.data_collator import DataCollatorWithPadding
from src.utils.cache_util import BufferedJsonWriter,BufferedJsonReader

from accelerate import Accelerator, DistributedType
import glob
import logging
logger = logging.getLogger(__name__)





class Scorer:
    def __init__(self,cfg, accelerator) -> None:
        print('cfg:\n{}'.format(cfg))
        print('cfg.dataset_reader:\n{}'.format(cfg.dataset_reader))

        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.dataset_reader.shard(accelerator)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer,device=accelerator.device)
        self.dataloader = DataLoader(self.dataset_reader,batch_size=cfg.batch_size,collate_fn=co)
        cfg.model.pretrained_model_name_or_path="./EleutherAI/gpt-neo-2.7B/"

        self.model = hu.instantiate(cfg.model)
        logger.info('self.scorer pretrained model type:{}'.format(type(self.model)))
        self.output_file = cfg.output_file
        self.accelerator = accelerator
        
        self.model = self.model.to(self.accelerator.device)
        self.model = self.model.eval()
        self.cfg = cfg
        self.input_history = []


    def forward(self):
        
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
        with BufferedJsonWriter(f"{self.output_file}tmp_{self.accelerator.device}.bin") as buffer:
            for i,entry in enumerate(dataloader):
                if "stop" in self.cfg and self.cfg.stop==i:
                    break
                metadata = entry.pop("metadata")
                with torch.no_grad():
                    output = self.model(input_ids=entry.input_ids,attention_mask=entry.attention_mask) 


                pad_mask = entry.pad_mask
                loss_list = sequence_cross_entropy_with_logits(logits=output.logits[:, :-1].contiguous(),
                                                               targets=entry.input_ids[:, 1:].contiguous(),
                                                               weights=pad_mask,
                                                               average=None)
                if len(loss_list.shape)==0:
                    loss_list = loss_list.unsqueeze(0)
                for mdata, loss in zip(metadata,loss_list):
                    mdata['score'] = float(loss.item())
                buffer.write(metadata)

        
    def write_results(self):
        def split_example(entry):
            test_example = {}
            train_example = {}
            for key,val in entry.items():
                if key.startswith("test_"):
                    test_example[key[len("test_"):]] = val
                else:
                    train_example[key] = val
            return test_example,train_example
        example_dict = {}
        data = []
        
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            logger.info('gather output_result from \"{}\"'.format(path))
            with BufferedJsonReader(path) as f:
                for x in f.read():
                    data.extend(x) 
        question_field = self.dataset_reader.task.question_field
        test_question_field = f"test_{question_field}"
        for entry in data:

            if entry[test_question_field] not in example_dict:
                test_example,train_example = split_example(entry)
                test_example['ctxs'] = [train_example]
                example_dict[entry[test_question_field]] = test_example
            else:
                _,train_example = split_example(entry)
                example_dict[entry[test_question_field]]['ctxs'].append(train_example)
        example_list = list(example_dict.values())
        if self.cfg.sort:
            for entry in example_list:
                question = entry.pop(question_field)
                entry['question'] = question
                entry['ctxs'] = sorted(entry['ctxs'],key = lambda x: x['score'])
        else:
            for entry in example_list:
                question = entry.pop(question_field)
                entry['question'] = question

        with open(self.output_file,"w") as f:
            json.dump(example_list,f)
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)


@hydra.main(config_path="configs",config_name="scorer")
def main(cfg):

    with open("cfg_scorer.json","w") as f:
        json.dump(OmegaConf.to_object(cfg),f)
    accelerator = Accelerator()
    scorer = Scorer(cfg, accelerator)

    scorer.forward()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        scorer.write_results()
    



if __name__ == "__main__":
    main()
    #import src
    #src.dataset_readers.scorer_dsr.ScorerDatasetReader