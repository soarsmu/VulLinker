import argparse
from multiprocessing import dummy
from django.shortcuts import render
import pickle
import pandas as pd
import sys 
import torch
import torch
from torch.utils.data import DataLoader


from deployed_model import model, tokenizer, label_map, inverse_label_map, MDataset


def home(request):
    return render(request, 'index.html')


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


def get_prediction(input_text: str) -> str:

    model.cuda()

    # dummy_text = "mozilla seamonkey firefox thunderbird mozilla firefox esr vulnerability class javascript engine mozilla firefox firefox thunderbird seamonkey attackers memory consumption garbage collection objects com errata rhsa html http www org security dsa http lists security announce msg00017 html http www com usn http www oracle com technetwork topics security html https bugzilla mozilla org show bug cgi http lists security announce msg00016 html http www mozilla security announce mfsa2014 html http archives html http lists security announce msg00022 html http rhn errata rhsa html http www securityfocus http lists security announce msg00016 html security gentoo glsa http www org security dsa firefox thunderbird"
    # dummy_text = "imagemagick attackers service segmentation fault application crash pnm file http com lists security https bugzilla show bug cgi http www openwall com lists security imagemagick"
    # dummy_text = "oracle jdk jre vulnerability oracle java se java se java se attackers confidentiality integrity availability vectors lists security announce msg00047 html http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http www securitytracker com http lists security announce msg00040 html http www org security dsa http lists security announce msg00039 html http www com usn security gentoo glsa http www oracle com technetwork topics security html http www securityfocus security gentoo glsa http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http www com usn http www org security dsa http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http rhn errata rhsa html http lists security announce msg00046 html http rhn errata rhsa html"
    # input_text = dummy_text # for debugging
    
    df = createOneData(text=input_text)

    max_len = 512
    batch = 1

    dataloader = DataLoader(MDataset(df, 'test', tokenizer, label_map, max_len),
                            batch_size=batch, num_workers=0,
                            shuffle=False)

    predicts = []
    predicts.append(torch.Tensor(model.one_epoch(
        0, dataloader, None, mode='test')[0]))

    prediction_labels = []

    for index, true_labels in enumerate(df.label.values):
        true_labels = set([label_map[i] for i in true_labels.split()])

        logits = [torch.sigmoid(predicts[index])]
        logits = [(-i).argsort()[:10].cpu().numpy() for i in logits]

        # print(logits[0][0][0])
        prediction_ids = logits[0][0][:3]

        for prediction_id in prediction_ids :
            prediction_labels.append(inverse_label_map[prediction_id])

    print("Input text: ", input_text)
    print("Predictions: ", prediction_labels)

    return prediction_labels   


# our result page view
def predict(request):
    name = request.GET['name']
    
    result = get_prediction(name)

    return render(request, 'predict.html', {'result': result})
