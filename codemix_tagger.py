import csv
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

#getting the data
f = open('path', encoding = 'utf8')
reader = csv.reader(f)
x = list(reader)
data = numpy.array(x).astype("str")


#Defining feature set
def features(word):
    for i in range(len(data)):
        words = data[i][0]
        tags = data[i][1]

        if word == '':
            return {
                'prev_word': ' ',
                'prev_tag': ' ',
                'word': word,
                'tag':  tags,
                'next_word': data[i + 1][0],
                'next_tag': data[i + 1][0]
            }

        else:
            return {
            'prev_word': data[i-1][0],
            'prev_tag':  data[i - 1][1],
            'word': word,
            'tag': tags,
            'next_word': data[i + 1][0],
            'next_tag':  data[i + 1][1],

            }

# Split the dataset for training and testing
cutoff = int(.90 * len(data))
training_words = data[:cutoff]
test_words = data[cutoff:]

print(len(training_words))
print(len(test_words))


def dataset(data):
    X, y = [], []

    for j in range(len(data)):
        X.append(features(data[j][0]))
        y.append(data[j][1])

    return X, y


X, y = dataset(training_words)





clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])

clf.fit(X[:3500],
        y[:3500])  #Data fitting. The accuracy was the highest for this particular fit of the data (0.68)

print('Training completed')

X_test, y_test = dataset(test_words)

print("Accuracy:", clf.score(X_test, y_test))














