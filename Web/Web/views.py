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
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


from deployed_model import model, tokenizer, label_map, inverse_label_map, MDataset
from crawling_reference import (crawl_bugs_launchpad, crawl_openwall, crawl_bugzilla_redhat, 
    crawl_access_redhat, crawl_rhn_redhat, crawl_lists_debian, crawl_debian, crawl_oracle, 
    crawl_lists_opensuse, crawl_fedora_pipermail, crawl_fedora_archives, crawl_security_gentoo, 
    crawl_security_gentoo_xml, crawl_security_gentoo_blogs, crawl_security_tracker, crawl_usn_ubuntu, 
    crawl_ubuntu, crawl_github)
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

def basic_cleaning(text):
    text = str(text)
    # Lowercase text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]','',text)
    # Remove redundant whitespaces
    text = ' '.join(text.split())
    return text


def nouns_only(text):
    tagged_text = nltk.tag.pos_tag(text.lower().split())
    editted_text = [word for word,tag in tagged_text if tag == 'NNP' or tag == 'NNPS' or tag=="NN" or tag=="NNS"]
    editted_text = " ".join(editted_text)
    editted_text = re.sub(r'[^\w\s]','',editted_text)
    editted_text = ' '.join(editted_text.split())
    return editted_text


def get_freq_dict(text):
    freq_dict = {}
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    for tok in text_tokens:
        if tok not in freq_dict:
            freq_dict[tok] = 1
        else:
            freq_dict[tok] += 1
    freq_dict = {k: v for k, v in sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)}
    return freq_dict

def frequent_words(text, proportion):
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    freq_dict = get_freq_dict(text)
    #round down
    num_words_to_remove = int(proportion * len(freq_dict))
    keys_to_remove = list(freq_dict.keys())[:num_words_to_remove]
    return keys_to_remove

def remove_frequent_words(text):
    try:
        text = re.sub(r'[^\w\s]','',text)
        word_tokens = word_tokenize(text.lower())
        filtered_sentence = []
        for w in word_tokens:
            if w not in freq_words:
                filtered_sentence.append(w)
        filtered_sentence = " ".join(filtered_sentence)
        filtered_sentence = re.sub(r'[^\w\s]','',filtered_sentence)
        filtered_sentence = ' '.join(filtered_sentence.split())
        return filtered_sentence
    except:
        return text

# our result page view
def predict(request):
    input_text = request.GET['input_text']
    
    result = get_prediction(nouns_only(basic_cleaning(input_text)))

    return render(request, 'predict.html', {'result': result})



def predict_by_cve_id(request):
    """make a prediction by using cve id. Get the vulnerability description from NVD database.
    Use https://github.com/vehemont/nvdlib to obtain the vulnerability description using the cve id from the NVD database.
    """
    cve_id = request.GET['cve_id']

    r = nvdlib.getCVE(cve_id)
        
    description = r.cve.description.description_data[0].value
    cpe = []
    cpe_text = ""
    config_cpe = r.configurations.nodes
    for eachNode in config_cpe:
        for eachCpe in eachNode.cpe_match:
            cpe.append(eachCpe.cpe23Uri)
            temp_cpe = eachCpe.cpe23Uri
            list_split = temp_cpe.split(":")
            cpe_text += list_split[3]
            cpe_text += " "
            cpe_text += list_split[4]
            cpe_text += " "
    reference_links = []
    for ref in r.cve.references.reference_data:
        reference_links.append(ref.url)
    reference_descs = []
    reference_link_text = ""
    for ref in reference_links:
        temp_ref = re.sub('[^0-9a-zA-Z]+', ' ', ref)
        reference_link_text += re.sub(' +', ' ', temp_ref)
        reference_link_text += " "
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
            if "usn.ubuntu" in short_ref:
                reference_descs.append(crawl_usn_ubuntu(ref))
            else:
                reference_descs.append(crawl_ubuntu(ref))
        else:
            reference_descs.append("")
    updated_desc = nouns_only(basic_cleaning(description))
    reference_text = ""
    for i_text in reference_descs:
        reference_text += nouns_only(basic_cleaning(i_text))
        reference_text += " "
    result=get_prediction(updated_desc+reference_link_text+cpe_text)


    return render(request, 'predict.html', {'result': result, 'cve_id': cve_id})
