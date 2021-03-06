from collections import Counter
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import scipy.sparse as sp
from math import log
import pickle as pkl
import tensorflow as tf
import os
from text_gcn.utils import *
from text_gcn.models import GCN, MLP
import time
from sklearn import metrics
from itertools import combinations

DATASET_PATH = "dataset/final_dataset.csv"

# get all columns from the pandas file which have the same label
def get_same_labels():
    description_fields = ["cve_id", "cleaned"]
    df = pd.read_csv(DATASET_PATH, usecols=description_fields)
    # df = pd.read_csv(DATASET_PATH, usecols=description_fields, nrows=100)
    # Read column names from file
    cols = list(pd.read_csv(DATASET_PATH, nrows=1))
    # print(cols)
    pd_labels = pd.read_csv(DATASET_PATH,
                            usecols=[i for i in cols if i not in ["cve_id", "cleaned", "matchers", "merged"]])

    # Get the list of available labels
    list_labels = [item for item in cols if item not in ["cve_id", "cleaned", "matchers", "merged"]]

    # Save to a string array too
    list_same_columns = []
    # Loop through all labels, find those that are duplicate
    for i in range(len(list_labels)):
        for j in range(i + 1, len(list_labels)):
            if pd_labels[list_labels[i]].equals(pd_labels[list_labels[j]]):
                list_same_columns.append(list_labels[i] + " == " + list_labels[j])
                print(list_labels[i] + " == " + list_labels[j])

    with open("SameLabels.txt", "w", encoding="utf-8") as file:
        for line in list_same_columns:
            file.write(line + "\n")

# Combine labels that always co-occur (i.e., a label that always occur together with other label)
def combine_labels():
    # read the csv
    df = pd.read_csv(DATASET_PATH)

    # Build the label dictionary
    label_dict = {}
    with open("same_labels.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().lstrip()
            labels = line.split(" == ")
            if labels[0] in label_dict:
                label_dict[labels[0]].append(labels[1])
            else:
                label_dict[labels[0]] = [labels[1]]
        f.close()

    # Build the name change dictionary (i.e., change between original name and the combined name)
    name_change_dictionary = {}

    print(label_dict)

    for key, arr in label_dict.items():
        # Check if the right hand side (i.e., the other label) exist in the column
        for value in arr:
            if value in df.columns:
                # print("INSIDE")
                # If exist, merge the other label name to the original label name
                if key in name_change_dictionary:
                    name_change_dictionary[key] = name_change_dictionary[key] + ";" + value
                else:
                    name_change_dictionary[key] = key + ";" + value
                # Then, drop the other label name
                df.drop(value, axis=1, inplace=True)

    # Then, change the column names based on name change dictionary
    df.rename(columns=name_change_dictionary, inplace=True)

    # Write to file
    df.to_csv("dataset/final_dataset_merged.csv")

combine_labels()