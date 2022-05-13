from helper import MDataset, loadCVEData
from model import LightXML

import torch 
from torch.utils.data import DataLoader



def get_exp_name():
    dataset = "cve_data"
    bert = ""
    name = [dataset, '' if bert == 'bert-base' else bert]

    return '_'.join([i for i in name if i != ''])


def get_inverse_label_map(label_map):
    inverse_label_map = {}
    for k, v in label_map.items():
        inverse_label_map[v] = k
    return inverse_label_map



df, label_map = loadCVEData()
inverse_label_map = get_inverse_label_map(label_map)

bert = "bert-base"
update_count = ""
swa = ""
swa_warmup = ""
swa_step = ""

global model
model = LightXML(n_labels=len(label_map), bert=bert,
                 update_count=update_count,
                 use_swa=swa, swa_warmup_epoch=swa_warmup, swa_update_step=swa_step)

model.load_state_dict(torch.load(
    f'../LightXML/models/model-{get_exp_name()}.bin'))
tokenizer = model.get_tokenizer()

print("MODEL")
print("MODEL")
print("MODEL")
print("MODEL")
print("MODEL")
# print(model)
