import keras
import pandas as pd
import gensim as gs
import numpy as np
import time
import os
from keras import Sequential
from keras.layers import Dense, Dropout, regularizers
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.utils import shuffle
from saveload import save
from sklearn.linear_model import LogisticRegression


def to_matrix(frame, vocabsize, vocab):
    freqs = {}
    matrixes = []
    strings = []
    tone = []
    for i in tqdm(range(frame.values.shape[0])):
        try:
            words = frame.iloc[i]['Lemmed'].split(' ')
            string = ""
            if len(words) == 0:
                break
            matr = np.zeros(shape=(1, vocabsize))
            for w in words:
                try:
                    matr = np.vstack((matr, vocab[w]))
                    string = string + ' ' + w
                    freqs[w] = freqs.get(w, 0) + 1
                except KeyError:
                    pass
            if matr.shape[0] > 2:
                matrixes.append(matr[1:matr.shape[0]][:])
                strings.append(string.strip())
                tone.append(frame.iloc[i]['Tone'])
        except AttributeError:
            print(frame.iloc[i]['Lemmed'])

    return matrixes, tone, strings, freqs


def pipeline(frame, vocabname):
    alpha = 0.0001
    n_epochs = 200
    n_parts = 1
    batch_size = 128
    sentence_embeddings = []

    vocab = gs.models.KeyedVectors.load_word2vec_format('models/' + vocabname + '.bin', binary=True)
    vocabsize = 300

    frame = shuffle(frame)
    matrixes, tones, strings, freqs = to_matrix(frame, vocabsize, vocab)
    num_words = sum(freqs.values())
    for key in freqs.keys():
        freqs[key] = freqs[key] / num_words

    for i in tqdm(range(len(matrixes))):
        embedding = np.zeros((vocabsize,))
        words = strings[i].split(' ')
        for j in range(len(words)):
            embedding += matrixes[i][j] * alpha / (alpha + freqs[words[j]])
        sentence_embeddings.append(embedding / len(words))

    del matrixes
    del strings

    sentence_embeddings = np.array(sentence_embeddings).T
    u, s, vh = np.linalg.svd(sentence_embeddings, full_matrices=False)
    fsv = np.array(u.T[0], ndmin=2)
    sm = np.matmul(fsv.T, fsv)
    sentence_embeddings = sentence_embeddings.T
    for i in range(sentence_embeddings.shape[0]):
        sentence_embeddings[i] -= np.matmul(sm, sentence_embeddings[i])

    tones = (np.array(tones) + 1) // 2

    train_embeddings = sentence_embeddings[0:sentence_embeddings.shape[0] * 4 // 5]
    train_tones = tones[0:sentence_embeddings.shape[0] * 4 // 5]

    test_embeddings = sentence_embeddings[sentence_embeddings.shape[0] * 4 // 5:]
    test_tones = tones[sentence_embeddings.shape[0] * 4 // 5:]

    params = {
        'C': np.logspace(-4, 1, 8),
        'penalty': ['l1', 'l2'],
        'dual': [False],
        'class_weight': [None, 'balanced']
    }
    grs = GridSearchCV(LogisticRegression(), params, verbose=2)

    # clf = SVC(kernel='sigmoid', class_weight='balanced', verbose=True)
    print('training')
    grs.fit(train_embeddings, train_tones)
    print(grs.best_score_)

    scores = grs.best_estimator_.score(train_embeddings, train_tones)
    print(scores)
    scores = grs.best_estimator_.score(test_embeddings, test_tones)
    print(scores)


if __name__ == '__main__':
    frame = pd.read_csv('shortened_united.csv', sep=';', header=0, names=['Text', 'Tone', 'Cleaned', 'Lemmed'])
    frame.dropna(inplace=True)

    pipeline(frame, 'news')
    pipeline(frame, 'wiki')
    pipeline(frame, 'web')
