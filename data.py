from scipy.sparse import data
import sklearn
import sklearn.feature_extraction.text
import sklearn.preprocessing
import sklearn.svm
import sklearn.decomposition

from collections import Counter
import random
import numpy
import re

def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

class Dataset(object):
    """
    Dataset object containing training data and test data
    """

    def __init__(self, file_path):
        # Read training data
        file = open(file_path)
        lines = file.readlines()
        data_in = []
        label_in = []
        is_label = False
        for line in lines:
            if is_label is True:
                left_paren_index = line.index("(")
                comma_index = line.index(",")
                right_paren_index = line.index(")")
                label_in.append(
                    (
                        line[0:left_paren_index],
                        line[left_paren_index + 1 : comma_index],
                        line[comma_index + 1 : right_paren_index],
                    )
                )
                is_label = False
            else:
                quote_index = line.index('"')
                data_in.append(clean_str(line[quote_index + 1 : -2]))
                is_label = True

        # divide into training data and test cases by 0.2 : 0.8
        (
            self.data_train,
            self.data_test,
            self.label_train,
            self.label_test,
        ) = sklearn.model_selection.train_test_split(
            data_in, label_in, test_size=0.1
        )
        # self.data_train = data_in
        # self.label_train = label_in

        label_count = Counter([label[0] for label in self.label_train])
        class_info = label_count.most_common()
        data_train_stat = {
            label: [] for label in label_count
        }
        for idx, data in enumerate(self.data_train):
            data_train_stat[self.label_train[idx][0]].append((idx, data))
        tmp_set = []
        max_label_count = class_info[0][1]
        for (label, label_count) in class_info:
            tmp_set += random.choices(data_train_stat[label], k=max_label_count-label_count)
        for idx, data in tmp_set:
            self.data_train.append(data)
            self.label_train.append(self.label_train[idx])

        self.raw_data_test = self.data_test

    def evaluate(self, model, sample_cnt=10):
        result = model.predict(self.data_test)
        right = 0
        wrong = []
        for i in range(len(self.label_test)):
            if result[i] == self.label_test[i]:
                right += 1
            else:
                wrong.append(i)

        print(right, len(self.label_test), right / len(self.label_test))
        for index in range(0, sample_cnt):
            i = wrong[index]
            print(self.raw_data_test[i])
            print("Wrong: {}, Cor: {}".format(self.label_decode(result[i]), self.label_decode(self.label_test[i])))

def print_matrix(real_label, predict_label):
    ret = numpy.zeros((10,10), dtype=numpy.float)
    for i in range(len(real_label)):
        ret[real_label[i]][predict_label[i]] += 1
    ret_sum = numpy.sum(ret, axis=1)
    return numpy.matmul(numpy.diag(1 / ret_sum), ret)

print_matrix([0,1,2,3,4,5,6,7,8,9,4,5,6,0,0,0,1], [0,2,2,3,4,5,6,7,8,9,6,5,4,0,1,0,0])