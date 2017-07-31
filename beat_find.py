import wave, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal


import parameters as P
import helpers as H


#Each window of data (either captured from the mic or read from the wav file) moves 256 samples at 22050Hz
time_step = 256.0/22050
instabilities = []
PATH = "closed/"

#When SIMULATE is set to true, input is from a wav file and output it a list of beat times
#When SIMULATE is set to false, input is from the microphone and output is a serial write at the beat times to the LED controller
SIMULATE = True

#If SIMULATE is set to true, plots may be optionally turned on to show some data for the song
DEBUG_PLOTS = False
def time_to_window_num(time):
    return int(time/time_step)-3

def window_num_to_time(idx):
    return (idx+3)*time_step

def main_thread(wav):
    found_beats = []
    #1. Init audio acquisition vars
    cur_sample, cur_window, start_window, cur_time, total_onset_power = 0,0,0,0,0
    onset_vecs = np.array([[], [], [], [], [], []], dtype=np.int)
    prev_onsets = np.zeros(6, dtype=np.int)
    time_vec = []
    band_confidence = [1] * 4
    #2. Init period finding vars
    tempo_instability = 0
    tempo_derivative = []
    tag_use_short_correlation, tag_use_med_correlation = False,False

    #3. Init beat finding vars
    prev_beat_guess, tentative_prev_time, prev_thresh, beat_thresh, beat_max = 0,0,0,0,0
    started_placing_beats, first_beat_selected, music_playing = False,False,True
    comb_pows, comb_times = [], []

    # 1. PERFORM AUDIO ACQUISITION
    sample_arr = np.fromstring(wav.readframes(2048), dtype='int16')[::2]

    while len(sample_arr) == 1024:

        # 2. CALCULATE TEMPO EVERY ~350ms
        recalc = False
        if cur_time > 4 and ((cur_window - start_window) % 30 == 0 or start_window == 0):
            recalc = True
            tempo_processing_thread(onset_vecs, cur_time)
            started_placing_beats = True
            start_window = cur_window

        cur_time = cur_sample / 44100
        time_vec.append(cur_time)

        windowed = np.hanning(1024) * sample_arr

        #Get the power spectrum
        x_psd, y_psd = scipy.signal.periodogram(windowed, 22050)
        y_psd = np.sqrt(y_psd)

        # Sum up the ranges
        onsets = np.array([0, 0, 0, 0, 0, 0])

        onsets[0] = np.sum(y_psd[0:P.P_FREQ_BAND_1])
        onsets[1] = np.sum(y_psd[P.P_FREQ_BAND_1:P.P_FREQ_BAND_2])
        onsets[2] = np.sum(y_psd[P.P_FREQ_BAND_2:P.P_FREQ_BAND_3])
        onsets[3] = np.sum(y_psd[P.P_FREQ_BAND_3:510])
        onsets[4] = onsets[3] + onsets[2] + onsets[1] + onsets[0]
    return found_beats, period_data

def grab_known(song_name):
    known_beats = []
    txtfile_name = PATH + song_name + ".txt"
    with open(txtfile_name) as f:
        for line in f:
            known_beats.append(float(line))
    known_pds = []
    for i in range(1, len(known_beats)):
        if known_beats[i] > 4:
            known_pds.append(known_beats[i] - known_beats[i-1])

    return known_beats, known_pds

def run_song(song_name, params=None):
    if params:
        H.set_params(params)
        P.FOURTH = params[-1]
    wav = wave.open(PATH + song_name + ".wav")
    wav.rewind()

    known_beats, known_pds = grab_known(song_name)
    #Run the algorithm
    main_thread(wav)

if __name__ == "__main__":
    run_song("closed_017")
