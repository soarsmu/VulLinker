import argparse
from multiprocessing import dummy
from django.shortcuts import render
import pickle
import tensorflow
import pandas as pd
import sys 
import torch

from helper import MDataset, loadCVEData
from  model import LightXML
from torch.utils.data import DataLoader

# sys.path.append("../LightXML/")

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=False, default=8)
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
parser.add_argument('--swa_step', type=int, required=False, default=200)

parser.add_argument('--group_y_group', type=int, default=0)
parser.add_argument('--group_y_candidate_num', type=int,
                    required=False, default=3000)
parser.add_argument('--group_y_candidate_topk',
                    type=int, required=False, default=10)

parser.add_argument('--eval_step', type=int, required=False, default=20000)

parser.add_argument('--hidden_dim', type=int, required=False, default=300)

parser.add_argument('--eval_model', action='store_true')

args = parser.parse_args()


def get_exp_name():
    name = [args.dataset, '' if args.bert == 'bert-base' else args.bert]
    if args.dataset in ['wiki500k', 'amazon670k']:
        name.append('t'+str(args.group_y_group))

    return '_'.join([i for i in name if i != ''])


def get_inverse_label_map(label_map):
    inverse_label_map = {}
    for k, v in label_map.items():
        inverse_label_map[v] = k
    return inverse_label_map


df, label_map = loadCVEData()
inverse_label_map = get_inverse_label_map(label_map)


model = LightXML(n_labels=len(label_map), bert=args.bert,
                 update_count=args.update_count,
                 use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step)

model.load_state_dict(torch.load(f'../../LightXML/models/model-{get_exp_name()}.bin'))
tokenizer = model.get_tokenizer()


def createOneData(text: str):
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
    for k, v in label_map.items():
        inverse_label_map[v] = k
    return inverse_label_map


def home(request):
    return render(request, 'index.html')


def get_prediction(name: str) -> str:

    # feature = count_vectorizer.transform([name])
    # probability = model.predict(feature)
    # prediction = ((probability) > 0.5).astype(int)

    # label = ['male' if p == MALE else 'female' for p in prediction]
    # return label[0]

    model = model.cuda()

    # dummy_text = "mozilla seamonkey firefox thunderbird mozilla firefox esr vulnerability class javascript engine mozilla firefox firefox thunderbird seamonkey attackers memory consumption garbage collection objects com errata rhsa html http www org security dsa http lists security announce msg00017 html http www com usn http www oracle com technetwork topics security html https bugzilla mozilla org show bug cgi http lists security announce msg00016 html http www mozilla security announce mfsa2014 html http archives html http lists security announce msg00022 html http rhn errata rhsa html http www securityfocus http lists security announce msg00016 html security gentoo glsa http www org security dsa firefox thunderbird"
    # dummy_text = "imagemagick attackers service segmentation fault application crash pnm file http com lists security https bugzilla show bug cgi http www openwall com lists security imagemagick"
    # dummy_text = "oracle jdk jre vulnerability oracle java se java se java se attackers confidentiality integrity availability vectors lists security announce msg00047 html http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http www securitytracker com http lists security announce msg00040 html http www org security dsa http lists security announce msg00039 html http www com usn security gentoo glsa http www oracle com technetwork topics security html http www securityfocus security gentoo glsa http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http www com usn http www org security dsa http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http lists security announce msg00046 html http rhn errata rhsa html"
    dummy_text = name
    df = createOneData(text=dummy_text)

    dataloader = DataLoader(MDataset(df, 'test', tokenizer, label_map, args.max_len),
                            batch_size=args.batch, num_workers=0,
                            shuffle=False)

    predicts = []
    predicts.append(torch.Tensor(model.one_epoch(
        0, dataloader, None, mode='test')[0]))

    prediction = ""

    for index, true_labels in enumerate(df.label.values):
        true_labels = set([label_map[i] for i in true_labels.split()])

        logits = [torch.sigmoid(predicts[index])]
        logits = [(-i).argsort()[:10].cpu().numpy() for i in logits]

        # print(logits[0][0][0])
        prediction = inverse_label_map[logits[0][0][0]]

    print("Input text: ", dummy_text)
    print("Prediction: ", prediction)

    return prediction




# our result page view
def predict(request):
    name = request.GET['name']
    
    result = get_prediction(name)

    return render(request, 'predict.html', {'result': result})
