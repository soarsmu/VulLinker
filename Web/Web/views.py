import argparse
from multiprocessing import dummy
from django.shortcuts import render
import pickle
import pandas as pd
import sys 
import torch
import torch
from torch.utils.data import DataLoader
from urllib.parse import urlparse


from deployed_model import model, tokenizer, label_map, inverse_label_map, MDataset
from crawling_reference import crawl_bugs_launchpad, crawl_openwall, crawl_bugzilla_redhat, crawl_access_redhat, crawl_rhn_redhat, crawl_lists_debian, crawl_debian, crawl_oracle, crawl_lists_opensuse, crawl_fedora_pipermail, crawl_fedora_archives, crawl_security_gentoo, crawl_security_gentoo_xml, crawl_security_gentoo_blogs, crawl_security_tracker, crawl_usn_ubuntu, crawl_ubuntu
import nvdlib


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
    input_text = request.GET['input_text']
    
    result = get_prediction(input_text)

    return render(request, 'predict.html', {'result': result})



def predict_by_cve_id(request):
    """make a prediction by using cve id. Get the vulnerability description from NVD database.
    Use https://github.com/vehemont/nvdlib to obtain the vulnerability description using the cve id from the NVD database.
    """
    cve_id = request.GET['cve_id']

    r = nvdlib.getCVE(cve_id)
        
    description = r.cve.description.description_data[0].value
    
    reference_links = []
    for ref in r.cve.references.reference_data:
        reference_links.append(ref.url)
    reference_descs = []
    for ref in reference_links:
        short_ref = urlparse(ref).netloc
        if "bugs.launchpad.net" in short_ref:
            reference_descs.append(crawl_bugs_launchpad(ref))
        elif "openwall.com" in short_ref:
            reference_descs.append(crawl_openwall(ref))
        elif "redhat.com" in short_ref:
            if "bugzilla" in short_ref:
                reference_descs.append(crawl_bugzilla_redhat(ref))
            elif "access" in short_ref:
                reference_descs.append(crawl_access_redhat(cve_id,ref))
            else:
                reference_descs.append(crawl_rhn_redhat(ref))
        elif "debian.org" in short_ref:
            if "lists" in short_ref:
                reference_descs.append(crawl_lists_debian(ref))
            else:
                reference_descs.append(crawl_debian(ref))
        elif "oracle.com" in short_ref:
            reference_descs.append(crawl_oracle(ref))
        elif "lists.opensuse.org" in short_ref:
            reference_descs.append(crawl_lists_opensuse(cve_id,ref))
        elif "fedoraproject.org" in short_ref:
            if "/pipermail/" in ref:
                reference_descs.append(crawl_fedora_pipermail(ref))
            elif "/archives/list/" in ref:
                reference_descs.append(crawl_fedora_archives(ref))
        elif "github" in short_ref:
            reference_descs.append(crawl_github(ref))
        elif "gentoo.org" in short_ref:
            if "security.gentoo.org/glsa/" in ref:
                reference_descs.append(crawl_security_gentoo_xml(ref))
            elif "blogs.gentoo.org/ago" in ref:
                reference_descs.append(crawl_security_gentoo_blogs(ref))
            elif "security.gentoo.org" in ref:
                reference_descs.append(crawl_security_gentoo(ref))
        elif "securitytracker" in short_ref:
            reference_descs.append(crawl_security_tracker(ref))
        elif "ubuntu" in short_ref:
            if "usn" in short_ref:
                reference_descs.append(crawl_usn_ubuntu(ref))
            else:
                reference_descs.append(crawl_ubuntu(ref))
        else:
            reference_descs.append("")

    result=get_prediction(description)

    return render(request, 'predict.html', {'result': result})
