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
from text_gcn.models import GCN
import time
from sklearn import metrics
from metrics import *

# DATASET_PATH = "dataset/final_dataset.csv"
DATASET_PATH = "dataset/final_dataset_merged.csv"

# Read the cleaned dataset csv file and split the dataset into train and test data
def data_preparation():
    description_fields = ["cve_id", "cleaned"]
    df = pd.read_csv(DATASET_PATH, usecols=description_fields, nrows=500)
    # df = pd.read_csv(DATASET_PATH, usecols=description_fields, nrows=100)
    # Read column names from file
    cols = list(pd.read_csv(DATASET_PATH, nrows=1))
    # print(cols)
    pd_labels = pd.read_csv(DATASET_PATH, usecols =[i for i in cols if i not in ["cve_id", "cleaned", "matchers", "merged"]], nrows=500)
    # pd_labels = pd.read_csv(DATASET_PATH, usecols =[i for i in cols if i not in ["cve_id", "cleaned", "matchers", "merged"]])

    # print(df.shape)
    # print(pd_labels.shape)
    data = df.to_numpy()
    labels = pd_labels.to_numpy()
    # Split dataset using skmultilearn (for multi-label classification)
    train, label_train, test, label_test = iterative_train_test_split(data, labels, test_size=0.2)
    # print("Train")
    # print(train)
    # print(label_train)
    # print("Test")
    # print(test)
    # print(label_test)
    return train, label_train, test, label_test

# build text gcn graph
# editted from the original text gcn repository
def build_graph():
    dataset = "nvd_data"
    dataset = "nvd_data_merged_500"
    word_embeddings_dim = 300
    word_vector_map = {}

    # Split the dataset using the data_preparation function
    train, label_train, test, label_test = data_preparation()

    # get the document name for the train, test, and overall data
    doc_train_list = train[:,0].tolist()
    doc_test_list = test[:,0].tolist()
    # IMPORTANT! Doc name list contain both train and test data. Train data first, continued with test data
    # I think this should be changed in the future to make it less error-prone
    doc_name_list = [*doc_train_list, *doc_test_list]
    label_dict = {}
    # Create a label dictionary
    # dictionary key is the document name (CVE ID)
    # dictionary value is the label
    for i in range(len(train)):
        label_dict[train[i][0]] = label_train[i]
    for i in range(len(test)):
        label_dict[test[i][0]] = label_test[i]


    # get the content (i.e., the description/feature) of the document considered in the dataset
    doc_content_list = train[:,1].tolist() + test[:,1].tolist()

    # print(len(doc_content_list))

    # get the id (index) of the train and test data
    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    # print("Train ID")
    # print(train_ids)
    # print(len(train_ids))
    random.Random(1).shuffle(train_ids)

    train_ids_str = '\n'.join(str(index) for index in train_ids)
    f = open('data/' + dataset + '.train.index', 'w')
    f.write(train_ids_str)
    f.close()

    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    random.Random(1).shuffle(test_ids)

    # print("Test ID")
    # print(test_ids)
    # print(len(test_ids))

    test_ids_str = '\n'.join(str(index) for index in test_ids)
    f = open('data/' + dataset + '.test.index', 'w')
    f.write(test_ids_str)
    f.close()

    ids = train_ids + test_ids

    print("Until here")
    # get the doc name list and doc words list that are shuffled
    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)].__str__())
        shuffle_doc_words_list.append(doc_content_list[int(id)].__str__())

    shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
    shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

    f = open('data/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_name_str)
    f.close()

    f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_words_str)
    f.close()

    # Build vocabulary, get the word frequency and list of word that appear in all dataset
    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_set)
    vocab_size = len(vocab)

    # print("Vocabulary")
    # print(vocab)
    # print(vocab_size)
    # print(word_freq)

    word_doc_list = {}
    # Not sure what this is for, compared to the vocabulary building above
    # The main difference is in the if word appeared then continue
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    # One hot encoding for the word mapping (i.e., mapping between a word to its id)
    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    vocab_str = '\n'.join(vocab)

    # print(word_id_map)
    f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
    f.write(vocab_str)
    f.close()

    # label list
    # list all the possible label from the dataset
    label_list = pd.read_csv(DATASET_PATH, nrows=0).columns.tolist()[4:]

    label_list_str = '\n'.join(label_list)
    f = open('data/corpus/' + dataset + '_labels.txt', 'w')
    f.write(label_list_str)
    f.close()

    # I think from here on is the feature creation process

    # Separate a validation dataset with size 10% of the training dataset
    # x: feature vectors of training docs, no initial features
    # slect 90% training set
    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size  # - int(0.5 * train_size)
    # different training rates

    real_train_doc_names = shuffle_doc_name_list[:real_train_size]
    real_train_doc_names_str = '\n'.join(real_train_doc_names)

    f = open('data/' + dataset + '.real_train.name', 'w')
    f.write(real_train_doc_names_str)
    f.close()

    print("Halfway there")
    # I think this is initialization process of the word embedding
    # The size of the word embeddings are:
    # Number of training data * Word Embedding dimension
    row_x = []
    col_x = []
    data_x = []
    for i in range(real_train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                # print(doc_vec)
                # print(np.array(word_vector))
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len
    # print("ROW COL DATA X")
    # print(len(row_x))
    # print(len(col_x))
    # print(len(data_x))

    # Initiate a Scipy compressed sparse row matrix
    # Shape would be (number of training data * embedding dimension)
    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
        real_train_size, word_embeddings_dim))

    y = []
    for i in range(real_train_size):
        doc_meta = shuffle_doc_name_list[i]
        y.append(label_dict[doc_meta])
    y = np.array(y)
    # print("This is Y")
    # print(y)
    # print(y.shape)


    # tx: feature vectors of test docs, no initial features
    # same as above but for test data
    test_size = len(test_ids)

    row_tx = []
    col_tx = []
    data_tx = []
    for i in range(test_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i + train_size]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_tx.append(i)
            col_tx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

    # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                       shape=(test_size, word_embeddings_dim))

    ty = []
    for i in range(test_size):
        doc_meta = shuffle_doc_name_list[i + train_size]
        ty.append(label_dict[doc_meta])
    ty = np.array(ty)

    # allx: the the feature vectors of both labeled and unlabeled training instances
    # (a superset of x)
    # unlabeled training instances -> words

    word_vectors = np.random.uniform(-0.01, 0.01,
                                     (vocab_size, word_embeddings_dim))

    for i in range(len(vocab)):
        word = vocab[i]
        # print(word)
        if word in word_vector_map:
            vector = word_vector_map[word]
            word_vectors[i] = vector

    # print(word_vectors)

    row_allx = []
    col_allx = []
    data_allx = []

    for i in range(train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))

    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)

    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

    ally = []

    for i in range(train_size):
        doc_meta = shuffle_doc_name_list[i]
        ally.append(label_dict[doc_meta])

    for i in range(vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)

    ally = np.array(ally)

    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    # word co-occurence with context windows
    window_size = 20
    windows = []

    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)

    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    row = []
    col = []
    weight = []

    # pmi as weights

    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)

    doc_word_freq = {}

    for doc_id in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))

    # dump objects
    f = open("data/ind.{}.x".format(dataset), 'wb')
    pkl.dump(x, f)
    f.close()

    f = open("data/ind.{}.y".format(dataset), 'wb')
    pkl.dump(y, f)
    f.close()

    f = open("data/ind.{}.tx".format(dataset), 'wb')
    pkl.dump(tx, f)
    f.close()

    f = open("data/ind.{}.ty".format(dataset), 'wb')
    pkl.dump(ty, f)
    f.close()

    f = open("data/ind.{}.allx".format(dataset), 'wb')
    pkl.dump(allx, f)
    f.close()

    f = open("data/ind.{}.ally".format(dataset), 'wb')
    pkl.dump(ally, f)
    f.close()

    f = open("data/ind.{}.adj".format(dataset), 'wb')
    pkl.dump(adj, f)
    f.close()



def training():
    dataset = "nvd_data_merged_full"
    # dataset = "nvd_data_merged_500"
    seed = 4
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # Settings
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('dataset', dataset, 'Dataset string.')
    # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_string('model', 'gcn', 'Model string.')
    flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 0,
                       'Weight for L2 loss on embedding matrix.')  # 5e-4
    flags.DEFINE_integer('early_stopping', 10,
                         'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
        FLAGS.dataset)
    print(adj)
    # print(adj[0], adj[1])
    features = sp.identity(features.shape[0])  # featureless

    print(adj.shape)
    print(features.shape)

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        # helper variable for sparse dropout
        'num_features_nonzero': tf.placeholder(tf.int32)
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=session_conf)

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(
            features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels, model.prediction_result, model.placeholderlabels], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test), outs_val[4], outs_val[5]

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(
            features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy,
                         model.layers[0].embedding, model.prediction_result, model.placeholderlabels], feed_dict=feed_dict)
        # print("K BEST PREDICTION")
        # prediction_list = get_k_best_prediction(3, outs[4])
        # count_correct_prediction = count_valid_prediction(prediction_list, outs[5])
        # print(count_correct_prediction)
        # Validation
        cost, acc, pred, labels, duration, prediction_list, label_list = evaluate(
            features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_f1=", "{:.5f}".format(
                outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_f1=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, pred, labels, test_duration, prediction_list, label_list = evaluate(
        features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "f1_score=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    test_pred = []
    test_labels = []


    for i in range(len(test_mask)):
        if test_mask[i]:
            test_pred.append(prediction_list[i])
            test_labels.append(label_list[i])

    test_pred = np.asarray(test_pred)
    test_labels = np.asarray(test_labels)

    # np.savetxt("test_pred.txt", test_pred)
    # np.savetxt("test_labels.txt", test_labels)

    actual_label = []
    for i in range(len(test_labels)):
        new_row = []
        for j in range (len(test_labels[i])):
            if test_labels[i][j] == 1:
                new_row.append(j)
        actual_label.append(new_row)
    label_array = np.asarray(actual_label)
    np.savetxt("actual_labels.txt", label_array, fmt='%s')

    print("PRINTING THE TEST LABELS AND TEST PREDICTION HERE")
    for i in range(1, 4):
        test_prediction = get_k_best_prediction(i, test_pred)
        prediction_array = np.asarray(test_prediction)
        np.savetxt("actual_prediction.txt", prediction_array)
        test_valid_prediction = count_valid_prediction(test_prediction, test_labels)
        test_precision = get_precision(test_valid_prediction, test_prediction)
        test_recall = get_recall(test_valid_prediction, test_labels)
        test_f1 = get_f1(test_precision, test_recall)
        print("Test Precision, Recall and F1-Score...")
        print("K = " + i.__str__())
        print("Precision: " + test_precision.__str__())
        print("Recall: " + test_recall.__str__())
        print("F1: " + test_f1.__str__())
        print()

    # print(metrics.classification_report(test_labels, test_pred, digits=4))
    # print("Macro average Test Precision, Recall and F1-Score...")
    # print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
    # print("Micro average Test Precision, Recall and F1-Score...")
    # print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))



# build_graph()
training()
# plot_label_distribution()
