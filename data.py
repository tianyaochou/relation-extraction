from scipy.sparse import data
import sklearn
import sklearn.feature_extraction.text
import sklearn.preprocessing
import sklearn.svm
import sklearn.decomposition

from collections import Counter
import random
import numpy

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
                data_in.append(line[quote_index + 1 : -2])
                is_label = True

        # divide into training data and test cases by 0.25 : 0.75
        (
            self.data_train,
            self.data_test,
            self.label_train,
            self.label_test,
        ) = sklearn.model_selection.train_test_split(
            data_in, label_in, test_size=0.25, random_state=0
        )

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