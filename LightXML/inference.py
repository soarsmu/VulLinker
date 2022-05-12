from lib2to3.pgen2.tokenize import tokenize
import sys
import random
import numpy as np
import pandas as pd
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


def createOneData(text:str):
    labels = []
    texts = []
    dataType = []


    texts.append(text.replace('\n', ''))
    dataType.append('test')
    labels.append('firefox')

    assert len(texts) == len(labels) == len(dataType)

    df_row = {'text': texts, 'label': labels, 'dataType': dataType}

    df = pd.DataFrame(df_row)

    return df


def get_inverse_label_map(label_map):
    inverse_label_map = {}
    for k, v in label_map.items() :
        inverse_label_map[v] = k
    return inverse_label_map


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
    
    inverse_label_map = get_inverse_label_map(label_map)
    
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

    dummy_text = "mozilla seamonkey firefox thunderbird mozilla firefox esr vulnerability class javascript engine mozilla firefox firefox thunderbird seamonkey attackers memory consumption garbage collection objects com errata rhsa html http www org security dsa http lists security announce msg00017 html http www com usn http www oracle com technetwork topics security html https bugzilla mozilla org show bug cgi http lists security announce msg00016 html http www mozilla security announce mfsa2014 html http archives html http lists security announce msg00022 html http rhn errata rhsa html http www securityfocus http lists security announce msg00016 html security gentoo glsa http www org security dsa firefox thunderbird"
    # dummy_text = "imagemagick attackers service segmentation fault application crash pnm file http com lists security https bugzilla show bug cgi http www openwall com lists security imagemagick"
    # dummy_text = "oracle jdk jre vulnerability oracle java se java se java se attackers confidentiality integrity availability vectors lists security announce msg00047 html http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http www securitytracker com http lists security announce msg00040 html http www org security dsa http lists security announce msg00039 html http www com usn security gentoo glsa http www oracle com technetwork topics security html http www securityfocus security gentoo glsa http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http www com usn http www org security dsa http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http lists security announce msg00046 html http rhn errata rhsa html"
    df = createOneData(text=dummy_text)

    dataloader = DataLoader(MDataset(df, 'test', tokenizer, label_map, args.max_len),
                            batch_size=args.batch, num_workers=0,
                            shuffle=False)
    
    predicts = []
    predicts.append(torch.Tensor(model.one_epoch(0, dataloader, None, mode='test')[0]))

    prediction = ""

    for index, true_labels in enumerate(df.label.values):
        true_labels = set([label_map[i] for i in true_labels.split()])

        logits = [torch.sigmoid(predicts[index])]
        logits = [(-i).argsort()[:10].cpu().numpy() for i in logits]

        # print(logits[0][0][0])
        prediction = inverse_label_map[logits[0][0][0]]

    print("Input text: ", dummy_text)
    print("Prediction: ", prediction)

    # python inference.py --dataset cve_data --swa --swa_warmup 10 --swa_step 200 --batch 8  

    
