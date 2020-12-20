from scipy.sparse import data
import sklearn
import sklearn.feature_extraction.text
import sklearn.preprocessing
import sklearn.svm
import sklearn.decomposition


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