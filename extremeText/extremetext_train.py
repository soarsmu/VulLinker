import extremeText
import re
from timeit import default_timer as timer

def print_results(N, p, r, c):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    print("C@{}\t{:.3f}".format(1, c))
    recall = 2 * p * r / (p + r)
    print("R@{}\t{:.3f}".format(1, recall))

def model_training():
    train_data = "dataset/train.txt"

    # train_supervised uses the same arguments and defaults as the fastText/extremeText cli

    print("Supervised Training")
    # default supervised training
    # model = extremeText.train_supervised(
    #     input=train_data, epoch=100, lr=1.0, wordNgrams=2, verbose=3, minCount=1,
    # )

    # paper supervised training
    model = extremeText.train_supervised(
        input=train_data, epoch=100, lr=0.05, verbose=3, wordNgrams=2, minCount=1, l2=0.003, arity=2, dim=100, tfidfWeights=True
    )
    model.save_model("model/xt_supervised.bin")

    # print("Quantization")
    #
    # model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
    #
    # model.save_model("model/xt_quantized.ftz")
#

def model_testing():
    test_data = "dataset/test_single"
    model_supervised = extremeText.load_model("model/xt_supervised.bin")
    # model_quantized = extremeText.load_model("model/xt_quantized.ftz")
    #
    # # Not sure if the testing code use the correct formula
    # # So below, we use our own using the prediction from the model
    # print("Test on supervised model")
    # print("K = 1")
    # print_results(*model_supervised.test(test_data, 1))
    # print("K = 2")
    # print_results(*model_supervised.test(test_data, 2))
    # print("K = 3")
    # print_results(*model_supervised.test(test_data, 3))
    #
    # print("Test on quantized model")
    # print("K = 1")
    # print_results(*model_quantized.test(test_data, 1))
    # print("K = 2")
    # print_results(*model_quantized.test(test_data, 2))
    # print("K = 3")
    # print_results(*model_quantized.test(test_data, 3))

    # Our own calculation
    sum_recall_1 = 0
    sum_recall_2 = 0
    sum_recall_3 = 0
    sum_precision_1 = 0
    sum_precision_2 = 0
    sum_precision_3 = 0

    num_test_data = 0
    with open(test_data, "r", encoding="utf-8") as w:
        content = w.readlines()
        num_test_data = len(content)
        for line in content:
            # get the labels from the line using regex
            labels = re.findall(r'\b__label__\S*', line)
            # sort the label list
            labels.sort(key=len, reverse=True)
            test_text = line
            # remove the labels from the text
            for label in labels:
                test_text = test_text.replace(label, "")
            test_text = test_text.lstrip().rstrip()
            prediction = model_supervised.predict(test_text, 3)
            correct_prediction = 0
            # Count the number of correct prediction for k = 1,2,3
            # K = 1
            if prediction[0][0] in labels:
                correct_prediction += 1
            sum_precision_1 += (correct_prediction / 1)
            sum_recall_1 += (correct_prediction / len(labels))

            # K = 2
            if prediction[0][1] in labels:
                correct_prediction += 1
            sum_precision_2 += (correct_prediction / 2)
            sum_recall_2 += (correct_prediction / len(labels))

            # K = 3
            if prediction[0][2] in labels:
                correct_prediction += 1
            sum_precision_3 += (correct_prediction / 3)
            sum_recall_3 += (correct_prediction / len(labels))

    # Divide by numer of test data
    precision_1 = sum_precision_1 / num_test_data
    precision_2 = sum_precision_2 / num_test_data
    precision_3 = sum_precision_3 / num_test_data
    recall_1 = sum_recall_1 / num_test_data
    recall_2 = sum_recall_2 / num_test_data
    recall_3 = sum_recall_3 / num_test_data

    f1_1 = 2
    f1_2 = 2
    f1_3 = 2
    # f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
    # f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2)
    # f1_3 = 2 * precision_3 * recall_3 / (precision_3 + recall_3)

    print("K = 1")
    print("P@1 = " + precision_1.__str__())
    print("R@1 = " + recall_1.__str__())
    print("F@1 = " + f1_1.__str__())

    print("K = 2")
    print("P@2 = " + precision_2.__str__())
    print("R@2 = " + recall_2.__str__())
    print("F@2 = " + f1_2.__str__())

    print("K = 3")
    print("P@3 = " + precision_3.__str__())
    print("R@3 = " + recall_3.__str__())
    print("F@3 = " + f1_3.__str__())


#
if __name__ == '__main__':
    start = timer()
    # model_training()

    model_testing()
    end = timer()
    print(end - start)