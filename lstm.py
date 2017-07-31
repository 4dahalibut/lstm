import scipy.io.wavfile
from sklearn.cross_validation import train_test_split
import random
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import scipy.signal
import matplotlib.pyplot as plt

REDO_ANNOTATIONS = False
start_time = time.time()
logs_path = '/home/josh/Documents/beats/lstm/logs/beatslog'
writer = tf.summary.FileWriter(logs_path)
test_len = 10
MAX_LEN = 5168
CONVERSION = 256/44100
ratio = 416/5168

def window2sec(wnum):
    return (wnum + 3) * CONVERSION

def sec2window(t):
    return int(t / CONVERSION)

def preprocess(wav):
    onsets = []
    sample_arr = wav[:1024]
    while len(sample_arr) == 1024:
        windowed = np.hanning(1024) * sample_arr
        _, psd = scipy.signal.periodogram(windowed, 22050)
        psd = np.sqrt(psd)

        onsets0 = np.sum(psd[0:7])
        onsets1 = np.sum(psd[7:26])
        onsets2 = np.sum(psd[26:188])
        onsets3 = np.sum(psd[188:510])
        onset = onsets0 + onsets1 + onsets2 + onsets3
        if len(onsets) == 0:
            onsets = [onset / 3]
        else:
            onsets.append((onset - onsets[-1]).clip(min=0))
        wav = wav[256:]
        sample_arr = wav[:1024]
    if len(onsets) > MAX_LEN:
        onsets = onsets[:MAX_LEN]
    onsets = np.array(onsets, dtype='int16')
    onsets = np.pad(onsets, (0, MAX_LEN - len(onsets)), 'constant', constant_values=-1)
    return np.reshape(onsets, (MAX_LEN, 1)), len(onsets)

def annotations(song_name, lens):
    correct = np.zeros((MAX_LEN), dtype=bool)
    known_beats = []
    txtfile_name = song_name + ".txt"
    last = 0
    with open(txtfile_name) as f:
        for line in f:
            known_beats.append(sec2window((float(line) + last) / 2))
            known_beats.append(sec2window(float(line)))
            last = float(line)
    freedom = int(.06 * np.mean(np.diff(known_beats)))
    for k in known_beats:
        for i in range(max(k - freedom, 0), min(k + freedom, lens - 1)):
            correct[i] = True
    return correct

def grab_wav(song_name):
    wav_raw = scipy.io.wavfile.read(song_name + ".wav")[1]
    onsets, lens = preprocess(wav_raw)
    y = annotations(song_name, lens)
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

class Iterator():
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
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_graph(state_size = 100, num_classes = 2, batch_size = 3):
    reset_graph()
    x = tf.placeholder(tf.float32, [batch_size, MAX_LEN, 1])
    seqlen = tf.placeholder(tf.int16, [batch_size])
    y = tf.placeholder(tf.int64, [batch_size, MAX_LEN])
    keep_prob = tf.placeholder_with_default(1.0, [])
    cell = tf.contrib.rnn.GRUCell(state_size)
    init_state = tf.get_variable('init_state', [1, state_size], initializer = tf.constant_initializer(0.0))
    init_state = tf.tile(init_state, [batch_size, 1])
    output, final_states = tf.nn.dynamic_rnn(cell, x, sequence_length=seqlen, initial_state = init_state)
    output = tf.nn.dropout(output, keep_prob)
    weight = tf.get_variable('weight', [state_size, num_classes])
    bias = tf.get_variable('bias', [num_classes], initializer=tf.constant_initializer(0.0))
    output = tf.reshape(output, [-1, state_size])
    pred = tf.nn.softmax(tf.matmul(output, weight) + bias, name="pred")
    pred_batch = tf.reshape(pred, [-1, MAX_LEN, num_classes])
    p = tf.argmax(pred_batch, 2)
    yb = tf.cast(y, tf.bool)
    y_not = np.invert(yb)
    #hits = tf.boolean_mask(p, yb)
    #mistakes = tf.boolean_mask(p, y_not)
    #accuracy = tf.reduce_mean(tf.cast(tf.equal(p, y), tf.float32))
    accuracy = tf.reduce_sum(tf.boolean_mask(p, yb)) / (MAX_LEN * batch_size)
    #a = tf.cast(y_not, tf.float32)
    #b = tf.log(prediction[:,:,0])
    class_weight = tf.constant([[ratio, 1.0-ratio]])
    labels = tf.one_hot(tf.reshape(y, [-1]), 2, dtype=tf.float32)
    weight_per_label = tf.transpose( tf.matmul(labels, tf.transpose(class_weight)) ) #shape [1, batch_size]
    xent = tf.multiply(weight_per_label, tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels, name="xent_raw"))
    loss = tf.reduce_mean(xent) #shape 1
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'accuracy': accuracy
    }

def train_graph(g, batch_size = 3, num_epochs = 30, iterator = Iterator):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data = getwavs()
        train, test = data[:-test_len], data[-test_len:]
        tr = Iterator(train)
        te = Iterator(test)

        step, accuracy = 0, 0
        tr_losses, te_losses = [], []
        current_epoch = 0
        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {g['x']: batch[0], g['y']: batch[2], g['seqlen']: batch[1], g['dropout']: 0.6}
            accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
            accuracy += accuracy_

            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(accuracy / step)
                step, accuracy = 0, 0

                #eval test set
                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    batch = te.next_batch(batch_size)
                    feed = {g['x']: batch[0], g['y']: batch[2], g['seqlen']: batch[1]}
                    accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
                    accuracy += accuracy_

                te_losses.append(accuracy / step)
                step, accuracy = 0,0
                print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])

g = build_graph()
train_graph(g)
