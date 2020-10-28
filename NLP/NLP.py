# model for understanding reviews (good/bad)


import matplotlib.pyplot as plt
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# quting is a param to specify how the quotes are gonna be processed
# the value 3 is used to skip all the quotes

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# cleaning the texts
corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)



