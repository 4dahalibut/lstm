import scipy.io.wavfile
import signal
import winsound
import random
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
import time
import scipy.signal
import matplotlib.pyplot as plt
import librosa.feature as rosa
import os

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
REDO_ANNOTATIONS = False
np.set_printoptions(threshold=np.nan)
start_time = time.time()
beatslog = '/home/josh/Documents/beats/lstm/logs/beatslog'
logs_path = beatslog
writer = tf.summary.FileWriter(logs_path)
TEST_LEN = 10
MAX_LEN = 2584
FS = 44100
CONVERSION = 512/FS
RATIO = 0.027029732038
FEATURES = 39
loss = []
accuracy = []

def window2sec(wnum):
    return wnum * CONVERSION

def sec2window(t):
    return int(t / CONVERSION)
def preprocess(wav):
    wav = np.array(wav, dtype='int64')
    onsets = np.zeros((MAX_LEN, FEATURES))
    i = 0
    low = wav[:1024]
    med = wav[:2048]
    hi = wav[:4096]
    _mfccl = 0
    _mfccm = 0
    _mfcch = 0
    _d1l, _d1m, _d1h = 0,0,0
    while len(hi) == 4096 and i < MAX_LEN:
        mfccl = rosa.mfcc(y=low, sr=44100, n_mfcc=4, n_fft=1024)[:,1]
        mfccm = rosa.mfcc(y=med, sr=44100, n_mfcc=4, n_fft=2048)[:,2]
        mfcch = rosa.mfcc(y=hi, sr=44100, n_mfcc=4, n_fft=4096)[:,4]
        el = np.sqrt(np.mean(np.abs(low)**2))
        em = np.sqrt(np.mean(np.abs(med)**2))
        eh = np.sqrt(np.mean(np.abs(hi)**2))
        d1l = np.maximum(mfccl - _mfccl, 0)
        d1m = np.maximum(mfccm - _mfccm, 0)
        d1h = np.maximum(mfccm - _mfcch, 0)
        d2l = np.maximum(d1l - _d1l, 0)
        d2m = np.maximum(d1m - _d1m, 0)
        d2h = np.maximum(d1h - _d1h, 0)
        l = []
        for el in [mfccl, mfccm, mfcch, el, em, eh, d1l, d1m, d1h, d2l, d2m, d2h]:
            if isinstance(el, np.ndarray):
                for eel in el:
                    l.append(eel)
            else:
                l.append(el)
        onsets[i,:] = l
        i += 1
        wav = wav[512:]
        _mfccl = mfccl
        _mfccm = mfccm
        _mfcch = mfcch
        _d1l = d1l
        _d1m = d1m
        _d1h = d1h
        low = wav[:1024]
        med = wav[:2048]
        hi = wav[:4096]

    if len(onsets) > MAX_LEN:
        onsets = onsets[:MAX_LEN]
    lonsets = len(onsets)
    onsets = np.pad(onsets, (0, MAX_LEN - len(onsets)), 'constant', constant_values=-1)[:,:FEATURES]
    return onsets, lonsets

def scale(data):
    new = []
    onsets = np.array([x[0] for x in data])
    onsets -= onsets.mean(axis=(0,1), keepdims=True)
    onsets /= onsets.std(axis=(0,1), keepdims=True)
    for idx, d in enumerate(data):
        new.append((onsets[idx,:,:], data[idx][1], data[idx][2]))
    return new

def annotations(song_name, lens):
    correct = np.zeros(MAX_LEN, dtype=bool)
    known_beats = []
    txtfile_name = song_name + ".txt"
    last = 0
    with open(txtfile_name) as f:
        for line in f:
            known_beats.append(sec2window((float(line) + last) / 2))
            known_beats.append(sec2window(float(line)))
            last = float(line)
    freedom = int(.05 * np.mean(np.diff(known_beats)))
    for k in known_beats:
        for i in range(max(k - freedom, 0), min(k + freedom, lens - 1)):
            correct[i] = True
    return correct

def grab_wav(song_name):
    fs, wav_raw = scipy.io.wavfile.read(song_name + ".wav")
    onsets, lens = preprocess(wav_raw)
    y = annotations(song_name, lens)
#provide wiggling here and then return a list of tuples instead of just a tuple, and then the flattening in getwavs will make it into a single list of tuples again
#Wiggling:
#Cut into 10sec chunks
    return onsets, lens, y

def getwavs():
    PATH = 'closed/'
    if REDO_ANNOTATIONS:
        wavs = [grab_wav(PATH + f[:-4]) for f in listdir(PATH) if isfile(join(PATH, f)) and f[-4:] == '.wav']
        with open('annotations', 'wb') as fp:
            pickle.dump(wavs, fp)
    else:
        with open ('annotations', 'rb') as fp:
            wavs = pickle.load(fp)
    return wavs

class Iterator:
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.df)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df[self.cursor:self.cursor+n]
        self.cursor += n
        return list(map(list, zip(*res)))

def reset_graph():
    with tf.Session() as sess:
        if 'sess' in globals() and sess:
            sess.close()
    tf.reset_default_graph()

def build_graph(state_size = 100, num_classes = 2, batch_size = 3):
    reset_graph()
    x = tf.placeholder(tf.float32, [batch_size, MAX_LEN, FEATURES])
    seqlen = tf.placeholder(tf.int16, [batch_size])
    y = tf.placeholder(tf.int64, [batch_size, MAX_LEN])
    keep_prob = tf.placeholder_with_default(1.0, [])
    cell = tf.contrib.rnn.GRUCell(state_size)
    init_state = tf.get_variable('init_state', [1, state_size], initializer = tf.constant_initializer(0.0))
    init_state = tf.tile(init_state, [batch_size, 1])
    output, final_states = tf.nn.dynamic_rnn(cell, x, sequence_length=seqlen, initial_state = init_state)
    #output = tf.nn.dropout(output, keep_prob)
    weight = tf.get_variable('weight', [state_size, num_classes])
    bias = tf.get_variable('bias', [num_classes], initializer=tf.constant_initializer(0.0))
    output = tf.reshape(output, [-1, state_size])
    pred = tf.nn.softmax(tf.matmul(output, weight) + bias, name="pred")#[15502,2]
    pred_batch = tf.reshape(pred, [-1, MAX_LEN, num_classes])
    p = tf.argmax(pred_batch, 2)
    yb = tf.cast(y, tf.bool)
    y_not = np.invert(yb)
    accuracy = tf.reduce_sum(tf.boolean_mask(p, yb)) / (MAX_LEN * batch_size)
    class_weight = tf.constant([[RATIO, 1.0-RATIO]])
    labels = tf.one_hot(tf.reshape(y, [-1]), 2, dtype=tf.float32) #[15504, 2]
    mask = tf.reshape(tf.sequence_mask(seqlen,MAX_LEN, dtype=tf.float32), [-1])
    weight_per_label = tf.transpose( tf.matmul(labels, tf.transpose(class_weight)) ) #shape [1, 15502]]
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels)#[15502]
    xent = tf.multiply(weight_per_label, cross_entropy) * mask #[1,15502]
    loss = tf.reduce_mean(xent) / tf.reduce_mean(mask) #shape 1
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'accuracy': accuracy
    }

def train_graph(g, batch_size = 3, num_epochs = 30):
    global loss, accuracy
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data = scale(getwavs()) #This is a 58 element long list of 3-tuples, each with, I think a [time, features] size onset array
        train, test = data[:-TEST_LEN], data[-TEST_LEN:]
        tr = Iterator(train)
        te = Iterator(test)

        step, accuracy, loss = 0, 0, 0
        tr_losses, te_losses = [], []
        current_epoch = 0

        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {g['x']: batch[0], g['y']: batch[2], g['seqlen']: batch[1], g['dropout']: 0.6}
            accuracy_, a, loss_ = sess.run([g['accuracy'], g['ts'], g['loss']], feed_dict=feed)
            accuracy += accuracy_
            loss += loss_

            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(loss / step)
                loss, step, accuracy = 0, 0, 0

                #eval test set
                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    batch = te.next_batch(batch_size)
                    feed = {g['x']: batch[0], g['y']: batch[2], g['seqlen']: batch[1]}
                    loss_, accuracy_ = sess.run([g['loss'], g['accuracy']], feed_dict=feed)
                    accuracy += accuracy_
                    loss += loss_

                te_losses.append(loss / step)
                loss, step, accuracy = 0, 0, 0
                print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])

def plot():
    if loss:
        plt.plot(loss)
        plt.plot(accuracy)
        plt.show()

signal.signal(signal.SIGINT, plot)
signal.signal(signal.SIGTERM, plot)
g = build_graph()
train_graph(g)

