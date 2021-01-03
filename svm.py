import sklearn
import sklearn.feature_extraction.text
import sklearn.preprocessing
import sklearn.svm
import sklearn.decomposition

class Dataset(object):
    """
    Object holding dataset
    """
    def __init__(self, file_path):
        # Read training data
        file  = open(file_path)
        lines = file.readlines()
        data_in = []
        label_in = []
        is_label = False
        for line in lines:
            if is_label is True:
                para_index = line.index('(')
                label_in.append(line[0 : para_index])
                is_label = False
            else:
                quote_index = line.index('"')
                data_in.append(line[quote_index + 1 : -1])   
                is_label = True  
                
        self.featurizor = sklearn.feature_extraction.text.TfidfVectorizer()
        vec_data = self.featurizor.fit_transform(data_in)
        # PCA
        # self.pca = sklearn.decomposition.PCA(n_components=3000)
        # vec_data = self.pca.fit_transform(vec_data.toarray())

        self.label_encoder = sklearn.preprocessing.LabelEncoder()
        label_in = self.label_encoder.fit_transform(label_in)

        # divide into training data and test cases by 0.25 : 0.75
        self.data_train, self.data_test, self.label_train, self.label_test = sklearn.model_selection.train_test_split(vec_data, label_in, test_size=0.25, random_state=0)

    def train(self):
        self.svm = sklearn.svm.LinearSVC()
        self.svm.fit(self.data_train, self.label_train)

    def evaluate(self):
        result = self.svm.predict(self.data_test)
        right = 0
        for i in range(len(self.label_test)):
            if result[i] == self.label_test[i]:
                right += 1
        print(right, len(self.label_test), right / len(self.label_test))
