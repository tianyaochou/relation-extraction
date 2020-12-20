from functools import reduce
from operator import pos
from data import Dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.linear_model import LogisticRegression
from collections import Counter
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

data = Dataset("./train.txt")


def count_words(sentence):
    c = Counter()
    for word in pos_tag(word_tokenize(sentence)):
        # c[word[0] + "/" + word[1]] += 1
        c[word[0]] += 1
    return c


middles = []
for i in range(len(data.data_train)):
    middle_start = data.data_train[i].index(data.label_train[i][1])
    middle_end = data.data_train[i].index(data.label_train[i][2])
    if middle_start < middle_end:
        middle = data.data_train[i][
            middle_start + +len(data.label_train[i][1]) : middle_end
        ]
    else:
        middle = data.data_train[i][
            middle_end + len(data.label_train[i][2]) : middle_start
        ]
    # middle = data.data_train[i]
    middles.append(count_words(middle))

data.data_test = [count_words(sentence) for sentence in data.data_test]

featurizor = DictVectorizer()
label_encoder = LabelEncoder()
data.data_train = featurizor.fit_transform(middles)
data.data_test = featurizor.transform(data.data_test)
data.label_decode = lambda x: label_encoder.inverse_transform([x])[0]

data.label_train = label_encoder.fit_transform([label[0] for label in data.label_train])
data.label_test = label_encoder.transform([label[0] for label in data.label_test])

svm = LinearSVC()
svm.fit(data.data_train, data.label_train)

data.evaluate(svm)