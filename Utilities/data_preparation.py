import os
import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.model_selection import iterative_train_test_split
import json
import sklearn
import pickle

## use this for the original csv file from ICPC paper
# TRAIN_PATH = "dataset/dataset_train.csv"
# TEST_PATH = "dataset/dataset_test.csv"
# FEATURE_NAME = "merged"
# DESCRIPTION_FIELDS = ["cve_id", FEATURE_NAME]
NON_LABEL_COLUMNS = ["cve_id", "description_text", "cpe_text", "merged", "description_and_reference"]

## use this for the new combined description and reference data
TRAIN_PATH = "dataset/dataset_train_reference.csv" # please change the csv name later, I use zero shot to indicate that we get this csv file from Vulnerability_Report_Tool_Paper repo
TEST_PATH = "dataset/dataset_test_reference.csv" # please change the csv name later, I use zero shot to indicate that we get this csv file from Vulnerability_Report_Tool_Paper repo
FEATURE_NAME = "description_and_reference"
DESCRIPTION_FIELDS = ["cve_id", FEATURE_NAME]
# NON_LABEL_COLUMNS = ["cve_id", "cleaned", "matchers", "merged", "reference", "description_and_reference", "year"]


# According to the usual division, we divide the dataset into 0.75:0.25 between the training and test data
# the splitted result will be saved in the dataset/splitted folder into 4 different files:
# splitted_train_x = training data (description/reference/etc.)
# splitted_train_y = label for training data
# splitted_test_x = test data
# splitted_test_y = label for test data
def split_dataset():
    description_fields = DESCRIPTION_FIELDS
    # Initiate the dataframe containing the CVE ID and its description
    # Change the "merged" field in the description_fields variable to use other text feature such as reference
    df = pd.read_csv(TRAIN_PATH, dtype=str, usecols=description_fields)
    #df.description_and_reference = df.description_and_reference.astype(str)
    # Read column names from file
    cols = list(pd.read_csv(TRAIN_PATH, nrows=1))
    # Initiate the dataframe containing the labels for each CVE
    pd_labels = pd.read_csv(TRAIN_PATH,
                            usecols=[i for i in cols if i not in NON_LABEL_COLUMNS])
    # Initiate a list which contain the list of labels considered in te dataset
    list_labels = [i for i in cols if i not in NON_LABEL_COLUMNS]
    print(list_labels)
    # Convert to numpy for splitting
    train = df.to_numpy()
    label_train = pd_labels.to_numpy()
    
    df2 = pd.read_csv(TEST_PATH, dtype=str, usecols=description_fields)
    #df2.description_and_reference = df2.description_and_reference.astype(str)
    # Read column names from file
    cols2 = list(pd.read_csv(TEST_PATH, nrows=1))
    # Initiate the dataframe containing the labels for each CVE
    pd_labels2 = pd.read_csv(TEST_PATH,
                            usecols=[i for i in cols if i not in NON_LABEL_COLUMNS])
    # Initiate a list which contain the list of labels considered in te dataset
    list_labels2 = [i for i in cols if i not in NON_LABEL_COLUMNS]
    # Convert to numpy for splitting
    test = df2.to_numpy()
    label_test = pd_labels2.to_numpy()
    # print(df2)
    # print(df)
    # Splitting using skmultilearn iterative train test split
    #train, label_train, test, label_test = iterative_train_test_split(data, labels, test_size=0.25)
    # Save the splitted data to files
    np.save("dataset/splitted/splitted_train_x.npy", train, allow_pickle=True)
    np.save("dataset/splitted/splitted_train_y.npy", label_train, allow_pickle=True)
    np.save("dataset/splitted/splitted_test_x.npy", test, allow_pickle=True)
    np.save("dataset/splitted/splitted_test_y.npy", label_test, allow_pickle=True)


# the test and train data are the same with omikuji
# however, you need to create the train/test_labels.txt and train/test_texts.txt
# with each row contains the text and labels for the train/test data
def prepare_lightxml_dataset():
    # Load the splitted dataset files
    train = np.load("dataset/splitted/splitted_train_x.npy", allow_pickle=True)
    label_train = np.load("dataset/splitted/splitted_train_y.npy", allow_pickle=True)
    test = np.load("dataset/splitted/splitted_test_x.npy", allow_pickle=True)
    label_test = np.load("dataset/splitted/splitted_test_y.npy", allow_pickle=True)
    train_corpus = train[:, 1].tolist()
    test_corpus = test[:, 1].tolist()
    cols = list(pd.read_csv(TRAIN_PATH, nrows=1))
    label_columns = [i for i in cols if i not in NON_LABEL_COLUMNS]
    num_labels = len(label_columns)

    vectorizer = TfidfVectorizer().fit(train_corpus)

    idx_zero_train = np.argwhere(np.all(label_train[..., :] == 0, axis=0))
    idx_zero_test = np.argwhere(np.all(label_test[..., :] == 0, axis=0))

    train_X = vectorizer.transform(train_corpus)
    # train_Y = np.delete(label_train, idx_zero_train, axis=1)
    train_Y = label_train
    test_X = vectorizer.transform(test_corpus)
    # test_Y = np.delete(label_test, idx_zero_test, axis=1)
    test_Y = label_test

    num_features = len(vectorizer.get_feature_names())
    num_row_train = train_X.shape[0]
    num_row_test = test_X.shape[0]

    # Dump the standard svmlight file
    train_fpath = "dataset/lightxml/train.txt"
    test_fpath = "dataset/lightxml/test.txt"

    if not os.path.exists(os.path.dirname(train_fpath)) :
        os.makedirs(os.path.dirname(train_fpath))

    dump_svmlight_file(train_X, train_Y, train_fpath, multilabel=True)
    dump_svmlight_file(test_X, test_Y, test_fpath, multilabel=True)

    train_text = []
    train_label = []
    test_text = []
    test_label = []

    cve_labels = pd.read_csv("dataset/cve_labels_merged_cleaned.csv")


    train_data = pd.read_csv(TRAIN_PATH)
    # process the label and text here
    for index, row in train_data.iterrows():
        train_text.append(row[FEATURE_NAME].lstrip().rstrip())
        # for label below
        label = cve_labels[cve_labels["cve_id"] == row.cve_id]
        label_unsplit = label.labels.values[0]
        label_array = label_unsplit.split(",")
        label_string = ""
        for label in label_array:
            label_string = label_string + label + " "
        label_string = label_string.rstrip()
        # print(label_string)
        train_label.append(label_string)

    test_data = pd.read_csv(TEST_PATH)
    for index, row in test_data.iterrows():
        test_text.append(row[FEATURE_NAME].lstrip().rstrip())
        # for label below
        label = cve_labels[cve_labels["cve_id"] == row.cve_id]
        label_unsplit = label.labels.values[0]
        label_array = label_unsplit.split(",")
        label_string = ""
        for label in label_array:
            label_string = label_string + label + " "
        label_string = label_string.rstrip()
        # print(label_string)
        test_label.append(label_string)


    with open("dataset/lightxml/train_texts.txt", "w", encoding="utf-8") as wr:
        for line in train_text:
            wr.write(line + "\n")

    with open("dataset/lightxml/train_labels.txt", "w", encoding="utf-8") as wr:
        for line in train_label:
            wr.write(line + "\n")

    with open("dataset/lightxml/test_texts.txt", "w", encoding="utf-8") as wr:
        for line in test_text:
            wr.write(line + "\n")

    with open("dataset/lightxml/test_labels.txt", "w", encoding="utf-8") as wr:
        for line in test_label:
            wr.write(line + "\n")

# Extreme text dataset should be separated into training data and test data (validation data optional)
# The prepare_extreme_text_dataset will separate the cleaned dataset into train and test dataset in
# accordance to the format required by extreme_text, which are:
# __label__LIBRARYNAME1 __label__LIBRARYNAME2 ... CVE_DESCRIPTION
# the label prefix can be set to using the label_prefix parameter,
# we use the default __label__ following the fasttext tutorial
# This function assume that the splitted dataset is available in dataset/splitted folder
def prepare_extreme_text_dataset(INPUT_DATASET, TRAINING_OUTPUT, TEST_OUTPUT, label_prefix = "__label__"):
    # Read column names from file
    cols = list(pd.read_csv(TRAIN_PATH, nrows=1))
    # Initiate the dataframe containing the labels for each CVE
    pd_labels = pd.read_csv(TRAIN_PATH,
                            usecols=[i for i in cols if i not in NON_LABEL_COLUMNS])
    # Initiate a list which contain the list of labels considered in te dataset
    list_labels = [i for i in cols if i not in NON_LABEL_COLUMNS]

    # Splitting using skmultilearn iterative train test split
    # train, label_train, test, label_test = iterative_train_test_split(data, labels, test_size=0.25)
    # Save the splitted data to files
    train_x = np.load("dataset/splitted/splitted_train_x.npy", allow_pickle=True)
    train_y = np.load("dataset/splitted/splitted_train_y.npy", allow_pickle=True)
    test_x = np.load("dataset/splitted/splitted_test_x.npy", allow_pickle=True)
    test_y = np.load("dataset/splitted/splitted_test_y.npy", allow_pickle=True)

    # process the training data to follow the extremetext requirements
    train_data = []
    # loop through all the training data
    for i in range(len(train_x)):
        text = train_x[i, 1]
        # get the index for the labels
        index_labels = np.nonzero(train_y[i])[0]
        labels = ""
        # loop through the label indexes, get the string representation and append to the string
        for idx in index_labels:
            labels = labels + label_prefix + list_labels[idx] + " "
        train_data.append(labels + text.lstrip())
    # write the train data to file
    with open(TRAINING_OUTPUT, "w", encoding="utf-8") as w:
        for line in train_data:
            w.write(line + "\n")
        w.close()

    # Do the same for the test data
    # process the test data to follow the extremetext requirements
    test_data = []
    # loop through all the training data
    for i in range(len(test_x)):
        text = test_x[i, 1]
        # get the index for the labels
        index_labels = np.nonzero(test_y[i])[0]
        labels = ""
        # loop through the label indexes, get the string representation and append to the string
        for idx in index_labels:
            labels = labels + label_prefix + list_labels[idx] + " "
        test_data.append(labels + text.lstrip())
    # write the train data to file
    with open(TEST_OUTPUT, "w", encoding="utf-8") as w:
        for line in test_data:
            w.write(line + "\n")
        w.close()

    # print(train_x[0])
    # print(train_y[0])

    # data = df.to_numpy()
    # labels = pd_labels.to_numpy()
    # # Split dataset using skmultilearn (for multi-label classification)
    # train, label_train, test, label_test = iterative_train_test_split(data, labels, test_size=test_size)
    # # print("Train")
    # # print(train)
    # # print(label_train)
    # # print("Test")
    # # print(test)
    # # print(label_test)
    # return train, label_train, test, label_test


# the dataset expected by the omikuji is the splitted dataset that has been extraced into BoW features
# while it is similar to the dataset required LightXML, which can be processed through sklearn dump_svmlight file
# and the TfIdfVectorizer, the omikuji dataset requires a header in the svmlight file
# the header consist of the space separated elements: <number of examples> <number of features> <number of labels>
# This function assume that the splitted dataset is available in dataset/splitted folder
def prepare_omikuji_dataset():
    # Load the splitted dataset files
    train = np.load("dataset/splitted/splitted_train_x.npy", allow_pickle=True)
    label_train = np.load("dataset/splitted/splitted_train_y.npy", allow_pickle=True)
    test = np.load("dataset/splitted/splitted_test_x.npy", allow_pickle=True)
    label_test = np.load("dataset/splitted/splitted_test_y.npy", allow_pickle=True)
    train_corpus = train[:, 1].tolist()
    test_corpus = test[:, 1].tolist()
    cols = list(pd.read_csv(TRAIN_PATH, nrows=1))
    label_columns = [i for i in cols if i not in NON_LABEL_COLUMNS]
    num_labels = len(label_columns)

    vectorizer = TfidfVectorizer().fit(train_corpus)

    idx_zero_train = np.argwhere(np.all(label_train[..., :] == 0, axis=0))
    idx_zero_test = np.argwhere(np.all(label_test[..., :] == 0, axis=0))



    train_X = vectorizer.transform(train_corpus)
    # train_Y = np.delete(label_train, idx_zero_train, axis=1)
    train_Y = label_train
    test_X = vectorizer.transform(test_corpus)
    # test_Y = np.delete(label_test, idx_zero_test, axis=1)
    test_Y = label_test

    num_features = len(vectorizer.get_feature_names())
    num_row_train = train_X.shape[0]
    num_row_test = test_X.shape[0]
    train_file_header = num_row_train.__str__() + " " + num_features.__str__() + " " + (num_labels).__str__()
    test_file_header = num_row_test.__str__() + " " + num_features.__str__() + " " + (num_labels).__str__()

    # Dump the standard svmlight file
    dump_svmlight_file(train_X, train_Y, "dataset/omikuji/train.txt", multilabel=True)
    dump_svmlight_file(test_X, test_Y, "dataset/omikuji/test.txt", multilabel=True)
    # Prepend the header to the svmlight file


    with open("dataset/omikuji/train.txt", 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(train_file_header.rstrip('\r\n') + '\n' + content)
        f.close()

    with open("dataset/omikuji/test.txt", 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(test_file_header.rstrip('\r\n') + '\n' + content)
        f.close()


def prepare_fastxml_dataset():
    # use the splitted dataset to create the train and test json required for the fastxml algorithm
    train = np.load("dataset/splitted/splitted_train_x.npy", allow_pickle=True)
    test = np.load("dataset/splitted/splitted_test_x.npy", allow_pickle=True)

    df_labels = pd.read_csv("dataset/cve_labels_merged_cleaned.csv", usecols=["cve_id", "labels"])
    # with open("dataset/train_non_iterative.json", "w") as f:
    with open("dataset/fastxml/train.json", "w") as f:
        for data in train:
            json_rep = {}
            json_rep["title"] = data[1].lstrip().rstrip()
            cve_id = data[0]
            cve_labels = df_labels[df_labels["cve_id"] == cve_id]["labels"].values.__str__()
            # Cleanup the label string from the cve_labels variable
            cve_labels = cve_labels.replace("[", "")
            cve_labels = cve_labels.replace("]", "")
            cve_labels = cve_labels.replace("'", "")
            cve_labels = cve_labels.replace('"', "")
            cve_labels = cve_labels.replace(" ", "")
            cve_labels = cve_labels.split(",")

            json_rep["tags"] = cve_labels
            json.dump(json_rep, f, ensure_ascii=False)
            f.write("\n")

    # with open("dataset/test_non_iterative.json", "w") as f:
    with open("dataset/fastxml/test.json", "w") as f:
        for data in test:
            json_rep = {}
            json_rep["title"] = data[1].lstrip().rstrip()
            cve_id = data[0]
            cve_labels = df_labels[df_labels["cve_id"] == cve_id]["labels"].values.__str__()
            # Cleanup the label string from the cve_labels variable
            cve_labels = cve_labels.replace("[", "")
            cve_labels = cve_labels.replace("]", "")
            cve_labels = cve_labels.replace("'", "")
            cve_labels = cve_labels.replace('"', "")
            cve_labels = cve_labels.replace(" ", "")
            cve_labels = cve_labels.split(",")
            json_rep["tags"] = cve_labels
            json.dump(json_rep, f, ensure_ascii=False)
            f.write("\n")

def prepare_xmlcnn_dataset_obsolete():
    train = np.load("dataset/splitted/splitted_train_x.npy", allow_pickle=True)
    label_train = np.load("dataset/splitted/splitted_train_y.npy", allow_pickle=True)
    test = np.load("dataset/splitted/splitted_test_x.npy", allow_pickle=True)
    label_test = np.load("dataset/splitted/splitted_test_y.npy", allow_pickle=True)
    # just need the text and the label number
    # for each entry, create a dictionary with the key: text and catgy
    # text contains the string / feature of the entry
    # catgy contains an array of labels numbers (int) related to the entry
    # prepare the train data first
    train_data = []
    for i in range(len(train)):
        text = train[i][1]
        label_sparse = label_train[i]
        label_ids = list(np.nonzero(label_sparse)[0])

        entry_dict = {}
        entry_dict["text"] = text
        entry_dict["catgy"] = label_ids
        train_data.append(entry_dict)

    # now prepare the test data
    test_data = []
    for i in range(len(test)):
        text = test[i][1]
        label_sparse = label_test[i]
        label_ids = list(np.nonzero(label_sparse)[0])

        entry_dict = {}
        entry_dict["text"] = text
        entry_dict["catgy"] = label_ids
        test_data.append(entry_dict)
    # dump the data using pickle
    with open("dataset/XML-CNN/train.pkl", "wb") as wt:
        pickle.dump(train_data, wt, pickle.HIGHEST_PROTOCOL)
    with open("dataset/XML-CNN/test.pkl", "wb") as wt:
        pickle.dump(test_data, wt, pickle.HIGHEST_PROTOCOL)

# The XML CNN dataset is in the form of tab-separated files (tsv) with three columns
# first column is the ID, an integer/String representing the ID of the entry
# second column is the labels
# third column is the text
# moreover, the XML CNN also require validation data, which by default is taken from the training data (25% of train data)
def prepare_xmlcnn_dataset():
    # load the splitted test train data
    train_unsplitted_x = np.load("dataset/splitted/splitted_train_x.npy", allow_pickle=True)
    train_unsplitted_y = np.load("dataset/splitted/splitted_train_y.npy", allow_pickle=True)
    test_x = np.load("dataset/splitted/splitted_test_x.npy", allow_pickle=True)
    test_y = np.load("dataset/splitted/splitted_test_y.npy", allow_pickle=True)

    # then re-split the train data to make the validation data
    train_x, train_y, valid_x, valid_y = iterative_train_test_split(train_unsplitted_x, train_unsplitted_y, test_size=0.25)

    # start with the train data
    with open("dataset/XML-CNN/train.txt", "w", encoding="utf-8") as w:
        for i in range(len(train_x)):
            id = train_x[i][0]
            text = train_x[i][1]
            labels = np.nonzero(train_y[i])[0]
            # initiate the entry with the id
            entry = id + "\t"
            # then the labels
            for label in labels:
                entry = entry + label.__str__() + " "
            # then the text, remove the last space from the label entry
            entry = entry[:-1] + "\t" + text
            # finally, write the entry to the file
            w.write(entry + "\n")

    # then the validation data
    with open("dataset/XML-CNN/valid.txt", "w", encoding="utf-8") as w:
        for i in range(len(valid_x)):
            id = valid_x[i][0]
            text = valid_x[i][1]
            labels = np.nonzero(valid_y[i])[0]
            # initiate the entry with the id
            entry = id + "\t"
            # then the labels
            for label in labels:
                entry = entry + label.__str__() + " "
            # then the text, remove the last space from the label entry
            entry = entry[:-1] + "\t" + text
            # finally, write the entry to the file
            w.write(entry + "\n")

    # finally, the test data
    with open("dataset/XML-CNN/test.txt", "w", encoding="utf-8") as w:
        for i in range(len(test_x)):
            id = test_x[i][0]
            text = test_x[i][1]
            labels = np.nonzero(test_y[i])[0]
            # initiate the entry with the id
            entry = id + "\t"
            # then the labels
            for label in labels:
                entry = entry + label.__str__() + " "
            # then the text, remove the last space from the label entry
            entry = entry[:-1] + "\t" + text
            # finally, write the entry to the file
            w.write(entry + "\n")
    print("XML CNN dataset created successfully!!!")



if __name__ == "__main__" :
    split_dataset()
    prepare_lightxml_dataset()
