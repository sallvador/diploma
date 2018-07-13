import keras
import pandas as pd
import gensim as gs
import numpy as np
import time
import os
from keras import Sequential
from keras.layers import Dense, Dropout, regularizers
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
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

    tones = np.array(tones)*0.5 + 0.5
    train_embeddings = sentence_embeddings[0:sentence_embeddings.shape[0] * 4 // 5]
    train_tones = tones[0:sentence_embeddings.shape[0] * 4 // 5]

    test_embeddings = sentence_embeddings[sentence_embeddings.shape[0] * 4 // 5:]
    test_tones = tones[sentence_embeddings.shape[0] * 4 // 5:]


    model_name = 'supermind'
    model = Sequential()
    model.add(Dense(2400, input_dim=300, activation='linear'))
    model.add(Dense(1, activation='sigmoid'))
    RMSpr = keras.optimizers.RMSprop(lr=0.0025)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSpr,
                  metrics=['accuracy'])

    times = []
    scores = []
    start_time = time.time()

    for i in range(n_epochs):
        model.fit(train_embeddings, train_tones, batch_size=batch_size, epochs=1, verbose=2, validation_data=(test_embeddings, test_tones))
        score = model.evaluate(test_embeddings, test_tones, batch_size=batch_size, verbose=1)
        scores.append(score)
        times.append(time.time() - start_time)
        print(score[1])

    path = os.path.abspath(os.getcwd()) + "/" + model_name + "_" + vocabname
    if not os.path.isdir(path):
        os.makedirs(path)
    model.save(path + "/model")
    save(scores, path, "/scores")
    save(times, path, "/times")


if __name__ == '__main__':
    frame = pd.read_csv('shortened_united.csv', sep=';', header=0, names=['Text', 'Tone', 'Cleaned', 'Lemmed'])
    frame.dropna(inplace=True)

    pipeline(frame, 'news')
    pipeline(frame, 'wiki')
    pipeline(frame, 'web')
