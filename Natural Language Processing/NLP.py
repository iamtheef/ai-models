# model for understanding reviews (good/bad)

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC


# quting is a param to specify how the quotes are gonna be processed
# the value 3 is used to skip all the quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# cleaning the texts
# stemming is done to crop all the words to their root (eg. "loved" to "love")
corpus = []
ps = PorterStemmer()
all_stopwords = stopwords.words('english')

# we are removing "not" from stopwords since we have to recognise negative reviews
all_stopwords.remove('not')

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
# print(corpus)
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# splitting the the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Training the kernel svm model to the training set
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the confusion matrix
print('confusion matrix : ', confusion_matrix(y_test, y_pred))
print('accuracy score : ', accuracy_score(y_test, y_pred))

