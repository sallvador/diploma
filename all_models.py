import pandas as pd
import numpy as np
import gensim as gs
from keras.models import Sequential
from keras.layers import LSTM, Flatten, Dense
from sklearn.utils import shuffle
from tqdm import tqdm
import time
import os
from saveload import save


def to_matrix(frame, vocabsize, vocab):
    matrixes = []
    tone = []
    for i in tqdm(range(frame.values.shape[0])):
        try:
            words = frame.iloc[i]['Lemmed'].split(' ')
            if len(words) == 0:
                break
            matr = np.zeros(shape=(1, vocabsize))
            for w in words:
                try:
                    matr = np.vstack((matr, vocab[w]))
                except KeyError:
                    pass
            if matr.shape[0] > 2:
                matrixes.append(matr[1:matr.shape[0]][:])
                tone.append(frame.iloc[i]['Tone'])
        except AttributeError:
            print(frame.iloc[i]['Lemmed'])

    return matrixes, tone


def simple_lstm(timesteps, vocabsize):
    model = Sequential()
    model.add(LSTM(20, return_sequences=True, input_shape=(timesteps, vocabsize), dropout=0.05))
    model.add(LSTM(1, return_sequences=False, dropout=0.05))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def lstm_with_mlp(timesteps, vocabsize):
    model = Sequential()
    model.add(LSTM(20, return_sequences=True, input_shape=(timesteps, vocabsize), dropout=0.05))
    model.add(LSTM(1, return_sequences=True, dropout=0.05))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def pipeline(frame, vocabname, model_type):
    vocab = gs.models.KeyedVectors.load_word2vec_format('models/' + vocabname + '.bin', binary=True)
    vocabsize = 300


    timesteps = 32
    batch_size = 64
    n_epochs = 200
    n_parts = 5


    model = None
    model_name = None
    if model_type==1:
        model = simple_lstm(timesteps, vocabsize)
        model_name = 'lstm_no_seq'
    elif model_type==2:
        model = lstm_with_mlp(timesteps, vocabsize)
        model_name = 'lstm_mlp'
    else:
        pass


    frame = shuffle(frame)
    matrices, tones = to_matrix(frame, vocabsize, vocab)
    tones = (np.array(tones) + 1) // 2
    for i in tqdm(range(len(matrices))):
        matrices[i] = np.vstack(
            (matrices[i], np.random.normal(scale=0.005, size=(timesteps - matrices[i].shape[0], vocabsize))))
    matrices = np.array(matrices)


    tones = np.array(tones, ndmin=2).T
    train_matrices = matrices[0:matrices.shape[0] * 4 // 5]
    train_tones = tones[0:matrices.shape[0] * 4 // 5]
    test_matrices = matrices[matrices.shape[0] * 4 // 5:]
    test_tones = tones[matrices.shape[0] * 4 // 5:]


    times = []
    scores = []
    start_time = time.time()


    for i in range(n_epochs):
        print('Epoch {}/{}'.format(i+1, n_epochs))
        for j in range(n_parts):
            matrices_heap = train_matrices[
                            j * train_matrices.shape[0] // n_parts:(j + 1) * train_matrices.shape[0] // n_parts]
            tones_heap = train_tones[
                         j * train_matrices.shape[0] // n_parts:(j + 1) * train_matrices.shape[0] // n_parts]
            model.fit(matrices_heap, tones_heap, batch_size=batch_size, epochs=1, verbose=2)
        if (i + 1) % 10 == 0:
            score = model.evaluate(test_matrices, test_tones, batch_size=batch_size, verbose=1)
            scores.append(score)
            times.append(time.time() - start_time)


    path = os.path.abspath(os.getcwd()) + "/" + model_name + "_" + vocabname
    if not os.path.isdir(path):
        os.makedirs(path)
    model.save(path + "/model")
    save(scores, path, "/scores")
    save(times, path, "/times")



if __name__ == '__main__':
    frame = pd.read_csv('shortened_united.csv', sep=';', header=0, names=['Text', 'Tone', 'Cleaned', 'Lemmed'])
    frame.dropna(inplace=True)
    pipeline(frame, 'news', 1)
    pipeline(frame, 'news', 2)
    pipeline(frame, 'wiki', 1)
    pipeline(frame, 'wiki', 2)
    pipeline(frame, 'web', 1)
