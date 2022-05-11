from lib2to3.pgen2.tokenize import tokenize
import sys
import random
import numpy as np
from apex import amp
from model import LightXML
import os

sys.path.append(".")
sys.path.append("./../")
sys.path.append(os.path.dirname(__file__))

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from transformers import AdamW

import torch

from torch.utils.data import DataLoader
from dataset import MDataset, createDataCSV
from log import Logger

def get_exp_name():
    name = [args.dataset, '' if args.bert == 'bert-base' else args.bert]
    if args.dataset in ['wiki500k', 'amazon670k']:
        name.append('t'+str(args.group_y_group))

    return '_'.join([i for i in name if i != ''])


def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=False, default=16)
parser.add_argument('--update_count', type=int, required=False, default=1)
parser.add_argument('--lr', type=float, required=False, default=0.0001)
parser.add_argument('--seed', type=int, required=False, default=6088)
parser.add_argument('--epoch', type=int, required=False, default=20)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--bert', type=str, required=False, default='bert-base')

parser.add_argument('--max_len', type=int, required=False, default=512)

parser.add_argument('--valid', action='store_true')

parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_warmup', type=int, required=False, default=10)
parser.add_argument('--swa_step', type=int, required=False, default=100)

parser.add_argument('--group_y_group', type=int, default=0)
parser.add_argument('--group_y_candidate_num', type=int, required=False, default=3000)
parser.add_argument('--group_y_candidate_topk', type=int, required=False, default=10)

parser.add_argument('--eval_step', type=int, required=False, default=20000)

parser.add_argument('--hidden_dim', type=int, required=False, default=300)

parser.add_argument('--eval_model', action='store_true')

args = parser.parse_args()


if __name__ == '__main__':
    init_seed(args.seed)

    print(get_exp_name())

    LOG = Logger('inference-log_' + get_exp_name())
    
    print(f'load {args.dataset} dataset...')
    print("Arguments)")
    print(args.dataset)
    print(os.getcwd())
    df, label_map = createDataCSV(args.dataset)
    
    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

    model = LightXML(n_labels=len(label_map), bert=args.bert,
                        update_count=args.update_count,
                        use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step)

    tokenizer = model.get_tokenizer()

    testloader = DataLoader(MDataset(df, 'test', tokenizer, label_map, args.max_len),
                            batch_size=args.batch, num_workers=0,
                            shuffle=False)

    validloader = DataLoader(MDataset(df, 'valid', tokenizer, label_map, args.max_len),
                                batch_size=args.batch, num_workers=0,
                                shuffle=False)


    print(f'load models/model-{get_exp_name()}.bin')
    model.load_state_dict(torch.load(f'models/model-{get_exp_name()}.bin'))
    model = model.cuda()

    print(len(df[df.dataType == 'test']))
    model.one_epoch(0, validloader, None, mode='eval')

    pred_scores, pred_labels = model.one_epoch(0, testloader, None, mode='test')
    np.save(f'results/{get_exp_name()}-labels.npy', np.array(pred_labels))
    np.save(f'results/{get_exp_name()}-scores.npy', np.array(pred_scores))

    # python inference.py --dataset cve_data --swa --swa_warmup 10 --swa_step 200 --batch 8  

    
