from numpy import logspace
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

if __name__ == '__main__':

    frame = pd.read_csv('shortened_united.csv', sep=';', header=0, names=['Text', 'Tone', 'Cleaned', 'Lemmed'])
    frame.dropna(inplace=True)
    frame = shuffle(frame)


    print('Doing tf-idf')
    tfidf = TfidfVectorizer(strip_accents='unicode')
    X = tfidf.fit_transform(frame['Lemmed'][:])

    params = {
        'C': logspace(-4, 1, 8),
        'penalty': ['l1'],
        'dual': [False]
    }
    gs = GridSearchCV(LinearSVC(), params, n_jobs=-1)

    print('training')
    gs.fit(X[0:4*X.shape[0]//5][:], frame['Tone'][0:4*X.shape[0]//5])
    print(gs.best_score_)
    print(dict(zip(np.unique(gs.best_estimator_.coef_, return_counts=True)))[0])
    print('testing')
    score = gs.best_estimator_.score(X[4*X.shape[0]//5:][:], frame['Tone'][4*X.shape[0]//5:])
    print(score)
